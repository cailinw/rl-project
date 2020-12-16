import sys
from nltk.translate.bleu_score import corpus_bleu

def main(candidate):
	references = []
	with open("/save/str_real_data.txt") as fin:
		for line in fin:
			line = line.split()
			references.append(line)

	candidates = []
	with open(candidate_file)as fin:
		for line in fin:
			line = line.split()
			candidates.append(line)

	scores = []
	for candidate in candidates:
		score = sentence_bleu(referencess, candidate)
		scores.append(score)

	print("BLEU Score: ", sum(scores) / len(scores))


if __name__ == "__main__":
	main(sys.argv[1])