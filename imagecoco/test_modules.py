import numpy as np
import torch

from rewarder import Rewarder, RewardModel


class Test:
	def __init__(self, batch_size, vocab_size):
		self.batch_size = batch_size
		self.vocab_size = vocab_size

	def test_reward_model(self):
		hidden_state_size = 64

		model = RewardModel(
			hidden_state_size = hidden_state_size,
			mlp_hidden_size = 128,
			embed_size = 100,
			vocab_size = self.vocab_size
		)

		x = torch.randn((self.batch_size, hidden_state_size))
		a = torch.randint(1, self.vocab_size, (self.batch_size,))
		
		result = model(x, a)
		print(result)
		print(result.shape)

	def test_rewarder(self):
		model = Rewarder()

	def runtests(self):
		self.test_reward_model()



if __name__=="__main__":
	test = Test(512, 4348)
	test.runtests()
