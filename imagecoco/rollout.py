import numpy as np
import torch

# TODO: Implement this
class Rollout():
	def __init__(self, model):
		self.model = model


	def generate(self):
		'''
		Return trajectories [seq_length x batch_size] and policy probs

		TODO: Implement this
		'''
		return [], []

	def get_reward(self, x, roll_num, rewarder):
		'''
		Compute reward for each partitial trajectory t:seq_length
		for all t 1:seq_length

		TODO: Implement this
		'''

		return []