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

        # https://huggingface.co/transformers/quickstart.html#using-the-past
        def generate(self, batch_size, num_batches, hidden_state):
                res = [] # for result

                # Put model in eval mode to disable dropout
                self.model.eval()

                # Generate. TODO hidden state
                for _ in num_batches:
                    batch = []
                    for _ in batch_size:
                        generated = ['<|endoftext|>']
                        context = torch.tensor([generated])
                        past = None

                        for _ in range(self.seq_len):
                            # TODO: get correct syntax
                            out, past = model(context=context, past_key_values=past)

                            # TODO: make stochastic
                            # need to map argmax -> token since cut out some vocab
                            token = self.token_map[torch.argmax(out[..., -1, :])]

                            generated += [token.tolist()]
                            context = token.unsqueeze(0)

                        batch.append(generated)

                    res.append(batch)

                # Returns tokens
                return res

        # Gets hidden state for inputted data (for rewards)
        def get_hidden_state(self, data_loader):
            self.model.eval()
            pass

        # fine tune new FC layer  using normal transformer opt & train data
        def pretrain_step(self, train_loader):
            self.model.train()
            pass
            

	def rl_train_step(self, x, rewards, policy_probs, decay_weight):
            # Put model in train mode
	    self.model.train()


if __name__ == '__main__':
    seq_len = 32
    vocab_size = 4839

    token_map = pickle.load(open('save/vocab_map.pkl', 'rb'))

    gen = Generator(seq_len, vocab_size, token_map)

    # TODO: train loop...
