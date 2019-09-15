import bcolz
import csv
import nltk
import pickle
import numpy as np

def get_glove_embedding(glove, word):
	if word in glove:
		return glove[word]
	elif word.lower() in glove:
		return glove[word.lower()]
	else:
		return np.zeros((100,))

def get_samples_from_tsv(filename):
	samples = []

	state_tag_labels = {
		"NONE": 0,
		"CREATE": 1,
		"DESTROY": 2,
		"MOVE": 3
	}

	seq_tag_labels = {
		"O": 0,
		"B-LOC-FROM": 1,
		"I-LOC-FROM": 2,
		"B-LOC-TO": 3,
		"I-LOC-TO": 4
	}

	with open(filename, "r") as fp:
		for line in fp:
			split = line.split()
			text = split[0].replace("############", "####,,,,,,,,,####,,,,,,,,,####")
			text = text.replace("########", "####,,,,,,,,,####")
			text = text.replace("####", " ")
			text_gloves = np.array([get_glove_embedding(glove, x) for x in text.split()])
			verb_tag = np.array([int(x) for x in split[1].split(",")])
			entity_tag = np.array([int(x) for x in split[2].split(",")])
			state_label = np.array([state_tag_labels[split[3]]])
			seq_label = np.array([seq_tag_labels[x] for x in split[4].split(",")])

			if text_gloves.shape[0] != verb_tag.shape[0] or text_gloves.shape[0] != entity_tag.shape[0]:
				print(line, text, text_gloves.shape, verb_tag.shape, entity_tag.shape)

			samples.append({
					"text": text,
					"text_gloves": text_gloves,
					"verb_tags": verb_tag,
					"entity_tags": entity_tag,
					"state_label": state_label,
					"seq_label": seq_label
				})

	return samples

# load glove dictionary

vectors = bcolz.open('data/6B.100.dat')[:]
words = pickle.load(open('data/6B.100_words.pkl', 'rb'))
word2idx = pickle.load(open('data/6B.100_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}

# generate samples from paragraphs
train_samples = get_samples_from_tsv("data/propara.run1.train.tsv")
test_samples = get_samples_from_tsv("data/propara.run1.test.tsv")
dev_samples = get_samples_from_tsv("data/propara.run1.dev.tsv")

with open("data/train_samples.pkl", "wb") as fp:
	pickle.dump(train_samples, fp)

with open("data/test_samples.pkl", "wb") as fp:
	pickle.dump(test_samples, fp)

with open("data/dev_samples.pkl", "wb") as fp:
	pickle.dump(dev_samples, fp)