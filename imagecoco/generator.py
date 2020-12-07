import numpy as np
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

# TODO: Implement this
class Generator():
	def __init__(self, seq_len, vocab_size, n_layer):
                self.seq_len = seq_len
                self.batch_size = batch_size

                self.model = GPT2LMHeadModel.from_pretrained('gpt2')
                self.model.lm_head = nn.Linear(self.model.lm_head..in_features, vocab_size)

                self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        def generate(self, batch_size, num_batches, hidden_state):
                res = [] # for result

                # Put model in eval mode to disable dropout
                self.model.eval()

                # Generate.
                for _ in num_batches:
                    batch = []
                    for _ in batch_size:
                        generated = self.tokenizer.encode('')
                        context = torch.tensor([generated])
                        past = None

                        for _ in range(self.seq_len):
                            out, past = model(context, past=past)
                            token = torch.argmax(out[..., -1, :])

                            generated += [token.tolist()]
                            context = token.unsqueeze(0)

                        batch.append(generated)

                    res.append(batch)
                # Returns ... .
                return res

	def rl_train_step(self, x, rewards, policy_probs, decay_weight):
                # Put model in train mode
		self.model.train()
