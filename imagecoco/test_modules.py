import numpy as np
import torch

from rewarder import Rewarder, RewardModel


class Test:
	def __init__(self, seq_length, batch_size, hidden_state_size, embed_dim, mlp_hidden_size):
		self.seq_length = seq_length
		self.batch_size = batch_size
		self.vocab_size = 4348
		self.hidden_state_size = hidden_state_size,
		self.embed_dim = embed_dim
		self.mlp_hidden_size = mlp_hidden_size
		self.learning_rate = 0.01

	def test_reward_model(self):
		hidden_state_size = 64

		model = RewardModel(
			hidden_state_size = hidden_state_size,
			mlp_hidden_size = self.mlp_hidden_size,
			embed_size = self.embed_dim,
			vocab_size = self.vocab_size
		)

		x = torch.randn((self.batch_size, hidden_state_size))
		a = torch.randint(1, self.vocab_size, (self.batch_size,))
		
		result = model(x, a)
		print(result)
		print(result.shape)

	def test_rewarder(self):
		model = Rewarder(self.seq_length, self.batch_size // 2, self.batch_size // 2, self.vocab_size, self.hidden_state_size, self.embed_dim, self.mlp_hidden_size, self.learning_rate)
		trajectories = torch.randn((self.batch_size, self.seq_length))
		rewarder.rewards_to_go(trajectories, 4)


	def runtests(self):
		self.test_reward_model()
		self.test_rewarder()



if __name__=="__main__":
	test = Test(32, 64, 512, 100, 128)
	test.runtests()
