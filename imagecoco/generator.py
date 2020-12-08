import numpy as np
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pickle
from torch.distributions import Categorical
import torch.nn.functional as F

class Generator():
	def __init__(self, seq_len, vocab_size, token_map):
                self.seq_len = seq_len
                self.batch_size = batch_size

                # declare our model, wanting to see hidden states
                self.model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True, use_cache=True)
                # mod head for our coco vocab
                self.model.lm_head = nn.Linear(self.model.lm_head.in_features, vocab_size)

                self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

                # due to changing the architecture, we need to map our argmax to the token
                # in the gpt2 tokenizer.
                self.token_map = np.array(token_map)

        def generate(self, batch_size, num_batches, inc_hidden_state, inc_probs):
            # put into eval mode
            self.model.eval()

            # create first word for all sentences, which is the start token
            generated = torch.tensor( (num_batches*batch_size)*[50256] )

            # tensor of probabilities
            probs = torch.empty(batch_size*num_batches, self.seq_len, vocab_size)

            # tensor of hidden states
            h_states = torch.empty(batch_size*num_batches, self.seq_len, self.model.config.n_embd)

            # for autoreg gen
            context = torch.tensor([generated])
            past = None

            # generate sequence
            for i in range(self.seq_len):
                prob, past, h_state = model(input_ids=context, past_key_values=past)

                # Attach hidden state (last layer)
                h_states[:, i, :] = h_state[-1].squeeze(1)

                # Get dist over tokens (softmax so sum to 1)
                prob = F.softmax(prob[..., -1, :], dim=1)
                # concat to probs array
                probs[:, i, :] = prob

                # Sample this prob dist for each sentence.
                dist = Categorical(prob)
                # Map this to the GPT2 token set.
                token = self.token_map[dist.sample()]

                # Add the new word to all the sentences
                generated = torch.cat([generated, token], dim=1)
                
                # get context ready for round 2
                context = token.unsqueeze(0)

            # split generated sentences into batches of size batch_size
            generated = torch.split(generated, batch_size, dim=0)

            res = [generated]
            if hidden_state:
                res += h_states
            if probs:
                res += probs

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
