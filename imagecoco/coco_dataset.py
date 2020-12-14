import pickle
from torch.utils.data import Dataset


class COCOImageCaptionsDataset(Dataset):
	"""COCO Image Captions dataset."""

	def __init__(self, pkl_file):
		"""
		Parameters:
			pkl_file (string): Path to the file with tokenized sentences.
		"""
		self.pkl_file = pkl_file
		self.data = pickle.load(open(pkl_file, "rb"))  # (num_sentences, m_in)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]
