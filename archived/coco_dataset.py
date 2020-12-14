import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class COCOImageCaptionsDataset(Dataset):
    """COCO Image Captions dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Parameters:
            file (string): Path to the file with tokenized sentences.
        """
        self.file = file
        sentences = []
		with open(data_file)as fin:
			for line in fin:
				line = line.strip()
				line = line.split()
				parse_line = [int(x) for x in line]
				sentences.append(parse_line)
		self.sentences = np.array(sentences)

	def __len__(self):
		return len(self.sentences)

	def __getitem__(self, idx):
		return self.sentences[idx]