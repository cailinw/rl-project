import numpy as np
import torch.nn as nn
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    LineByLineTextDataset,
)

# import pickle
from torch.distributions import Categorical
import torch.nn.functional as F


class Generator:
    def __init__(self, seq_len, token_map, str_map=None):
        self.seq_len = seq_len
        self.vocab_size = len(token_map)

        # declare our model, wanting to see hidden states
        self.model = GPT2LMHeadModel.from_pretrained(
            "gpt2", output_hidden_states=True, use_cache=True
        )

        # freeze transformer
        for param in self.model.transformer.parameters():
            param.requires_grad = False

        # mod head for our coco vocab
        self.model.lm_head = nn.Linear(self.model.lm_head.in_features, self.vocab_size)

        # Just making sure the FC layer is not frozen :)
        for param in self.model.lm_head.parameters():
            param.requires_grad = True

        # Use the same tok
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # due to changing the architecture, we need to map our argmax to the token
        # in the gpt2 tokenizer.
        self.token_map = torch.tensor(token_map)

        # map to map non-gpt vocab back into strings
        self.str_map = np.array(str_map)

    def generate(self, batch_size, num_batches, inc_hidden_state, inc_probs, decode):
        # put into eval mode
        self.model.eval()

        # placeholder for generated words
        generated = torch.empty(
            batch_size * num_batches, self.seq_len, dtype=torch.long
        )

        # tensor of probabilities
        probs = torch.empty(batch_size * num_batches, self.seq_len, self.vocab_size)

        # tensor of hidden states
        h_states = torch.empty(
            batch_size * num_batches, self.seq_len, self.model.config.n_embd
        )

        # start token
        tok = 50256 * torch.ones(batch_size * num_batches, dtype=torch.long)
        past = None

        # generate sequence
        for i in range(self.seq_len):
            # forward pass + extract data
            res = self.model(input_ids=tok, past_key_values=past)
            prob, past, h_state = res[0], res[1], res[2][-1]

            # pick out most recent token (if inputted > 1 token)
            if len(prob.shape) == 3:
                prob = prob[:, -1, :]
                h_state = h_state[:, -1, :]

            # Attach hidden state (last layer)
            h_states[:, i, :] = h_state.squeeze(1)

            # concat to probs array
            probs[:, i, :] = F.log_softmax(prob, dim=1)

            # Sample this prob dist for each sentence.
            dist = Categorical(F.softmax(prob, dim=1))

            # Add the new word to all the sentences (in non-gpt vocab)
            generated[:, i] = dist.sample()

            # map to gpt2 vocab
            tok = self.token_map[generated[:, i]]

        # decode=put back to string
        if decode:
            generated = self.str_map[generated.flatten()].reshape(
                batch_size * num_batches, self.seq_len
            )
        else:
            generated = np.split(np.array(generated), batch_size, axis=0)

        res = [generated]
        if inc_hidden_state:
            res.append(h_states)
        if inc_probs:
            res.append(probs)

        return res

    # Gets hidden state for inputted data (for rewards)
    def get_hidden_state(self, data):
        self.model.eval()

        # tokenize data
        tok = torch.tensor(self.tokenizer(data)["input_ids"])

        # pass thru transformer
        h_state = self.model(input_ids=tok)[2][-1]

        # pick out last token
        if len(h_state.shape) == 3:
            h_state = h_state[:, -1, :].squeeze(1)

        return h_state

    # fine tune new FC layer  using normal transformer opt & train data
    # https://huggingface.co/transformers/custom_datasets.html
    def pretrain(self, train_path):

        # Get train data
        train_dataset = LineByLineTextDataset(
            tokenizer=self.tokenizer, file_path=train_path, block_size=self.seq_len
        )

        # put model in train mode
        self.model.train()

        # train hyperparams
        training_args = TrainingArguments(
            output_dir="./results",  # output directory
            num_train_epochs=3,  # total number of training epochs
            per_device_train_batch_size=512,  # batch size per device during training
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir="./logs",  # directory for storing logs
            logging_steps=10,
        )

        # define trainer
        trainer = Trainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
        )

        trainer.train()  # train

    def rl_train_step(self, x, rewards_to_go, probs, decay_weight):
        # TODO: Can take in batch_size instead of 1?
        """
            Parameters
                x : (1, seq_length)
                rewards_to_go : (1, seq_length)
                probs : (1, seq_length, vocab_size)
            """

        # Put model in train mode
        self.model.train()

        # loss = 0
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

