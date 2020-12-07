import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

# TODO: Implement this
class Generator():
	def __init__(self, seq_len, vocab_size, n_layer):
                self.seq_len = seq_len
                self.batch_size = batch_size

                self.model = GPT2LMHeadModel(GPT2Config(vocab_size=4839, n_positions=seq_len, n_ctx=seq_len, \
                                                n_embd=128, n_layer=n_layer)).cuda()

        def generate(self, batch_size, num_batches, hidden_state):
                res = [] # for result

                # Put model in eval mode to disable dropout
                self.model.eval()

                # Generate.
                for _ in num_batches:
                    batch = []
                    for _ in batch_size:
                        generated = [0]
                        context = torch.tensor([generated])
                        past = None

                        for _ in range(self.seq_len):
                            out, past = model(context, past=past)
                            token = torch.argmax(out[..., -1, :])

                            generated += [token.tolist()]
                            context = token.unsqueeze(0)

                        batch.append(generated)

                    res.append(batch)

                return res

	def rl_train_step(self, x, rewards, policy_probs, decay_weight):
                # Put model in train mode
		self.model.train()
