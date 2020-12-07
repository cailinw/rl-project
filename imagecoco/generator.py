import numpy as np
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pickle

# TODO: Implement this
class Generator():
	def __init__(self, seq_len, vocab_size, token_map):
                self.seq_len = seq_len
                self.batch_size = batch_size

                self.model = GPT2LMHeadModel.from_pretrained('gpt2')
                # mod head for our coco vocab
                self.model.lm_head = nn.Linear(self.model.lm_head.in_features, vocab_size)

                self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

                # due to changing the architecture, we need to map our argmax to the token
                # in the gpt2 tokenizer.
                self.token_map = token_map

        def generate(self, batch_size, num_batches, hidden_state):
            res = [] # for result

            # put into eval mode
            self.model.eval()

            # generate
            res = self.model.generate(do_sample=True, num_return_sequences=num_batches*batch_size)

            # split into batches
            res = torch.split(res, batch_size, 0)

            # Returns tokens
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
