import numpy as np
import torch.nn as nn
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pickle
from torch.distributions import Categorical
import torch.nn.functional as F

class Generator():
        def __init__(self, seq_len, vocab_size, token_map):
                self.seq_len = seq_len
                self.vocab_size = vocab_size

                # declare our model, wanting to see hidden states
                self.model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True, use_cache=True)
                # mod head for our coco vocab
                self.model.lm_head = nn.Linear(self.model.lm_head.in_features, vocab_size)

                self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

                # due to changing the architecture, we need to map our argmax to the token
                # in the gpt2 tokenizer.
                self.token_map = torch.tensor(token_map)

        def generate(self, batch_size, num_batches, inc_hidden_state, inc_probs, decode):
            # put into eval mode
            self.model.eval()

            # placeholder for generated words
            generated = torch.empty(batch_size*num_batches, self.seq_len)

            # tensor of probabilities
            probs = torch.empty(batch_size*num_batches, self.seq_len, self.vocab_size)

            # tensor of hidden states
            h_states = torch.empty(batch_size*num_batches, self.seq_len, self.model.config.n_embd)

            # start token
            tok = 50256 * torch.ones(batch_size*num_batches, dtype=torch.long)
            past = None

            # generate sequence
            for i in range(self.seq_len):
                res = self.model(input_ids=tok, past_key_values=past)
                prob, past, h_state = res[0], res[1], res[2]

                # Attach hidden state (last layer)
                h_states[:, i, :] = h_state[-1].squeeze(1)

                # Get dist over tokens (softmax so sum to 1)
                prob = F.softmax(prob, dim=1)
                # concat to probs array
                probs[:, i, :] = prob

                # Sample this prob dist for each sentence.
                dist = Categorical(prob)
                # Map this to the GPT2 token set.
                tok = self.token_map[dist.sample()]

                # Add the new word to all the sentences
                generated[:, i] = tok

            if decode:
                str_gen = []
                for s in generated:
                    str_gen.append(self.tokenizer.decode(s))

                generated=str_gen
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
            pass

        # fine tune new FC layer  using normal transformer opt & train data
        def pretrain_step(self, data):


            self.model.train()
            pass
            

        def rl_train_step(self, data, rewards, policy_probs, decay_weight):
            # Put model in train mode
            self.model.train()
            pass
