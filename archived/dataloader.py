import numpy as np

class Dataloader():
	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.num_batches = 0
		self.sentences = np.array([])
		self.sentences_batches = np.array([])

	def shuffle_data(self):
		# Shuffle sentences
		shuffle_indices = np.random.permutation(np.arange(len(self.sentences)))
		self.sentences = self.sentences[shuffle_indices]


	def load_train_data(self, data_file):
		# Load sentences
		sentences = []
		with open(data_file)as fin:
			for line in fin:
				line = line.strip()
				line = line.split()
				parse_line = [int(x) for x in line]
				sentences.append(parse_line)
		self.sentences = np.array(sentences)

		self.shuffle_data()

		# Split batches
		self.num_batches = len(self.sentences) // self.batch_size		
		self.sentences_batches = np.split(self.sentences, self.num_batches, 0)
		self.pointer = 0

	def next_batch(self):
		texts = self.sentences_batches[self.pointer]
		self.pointer = (self.pointer + 1) % self.num_batches
		return texts

	def reset_pointer(self):
		self.pointer = 0