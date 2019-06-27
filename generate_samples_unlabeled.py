import bcolz
import csv
import nltk
import pickle
import numpy as np

from nltk.stem import WordNetLemmatizer

def get_samples_from_paras(paras):
	samples = []

	for para in paras:
		for i in range(len(para["texts_gloves"])):
			verb_tags = list(map(lambda x: 1. if x[1][0] == "V" else 0., nltk.pos_tag(para["texts"][i][:-1].split())))

			sample = {
				"gloves": np.array(para["texts_gloves"][i]),
				"text": para["texts"][i],
				"lemma_text": para["lemma_texts"][i],
				"num_words": len(para["texts"][i][:-1].split()),
				"participants": para["participants"],
				"verb_tags": np.array(verb_tags)
			}

			samples.append(sample)

	return samples


def get_glove_embedding(glove, word):
	if word in glove:
		return glove[word]
	elif word.lower() in glove:
		return glove[word.lower()]
	else:
		return np.zeros((100,))

# load glove dictionary

vectors = bcolz.open('data/6B.100.dat')[:]
words = pickle.load(open('data/6B.100_words.pkl', 'rb'))
word2idx = pickle.load(open('data/6B.100_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}

# load dataset

unlabeled_paras = []

lemmatizer = WordNetLemmatizer()

with open("data/ProPara_StateChange_Unlabeled.csv", newline='') as fp:
	csvReader = csv.reader(fp, delimiter=',')

	for row in csvReader:
		if row[1] == "SID":
			idx = int(row[0])
			texts = []
			lemma_texts = []
			participants = [x for x in row[3:] if x != "" and x != "\n"]
		elif row[2][:6] == "PROMPT":
			prompt = row[2][8:]
		elif row[1][:5] == "event":
			texts.append(row[2])
			lemma_texts.append([lemmatizer.lemmatize(x) for x in row[2][:-1].split()])
		if row[0] == "" and idx != -1:
			sample = {
				"idx": idx,
				"prompt": prompt,
				"texts": texts,
				"lemma_texts": lemma_texts,
				"participants": participants,
				"combined_text": " ".join(texts),
				"prompt_gloves": [get_glove_embedding(glove, x) for x in prompt[:-1].split()] + [glove["?"]],
				"texts_gloves": [[get_glove_embedding(glove, x) for x in lemma_text] for lemma_text in lemma_texts]
			}

			unlabeled_paras.append(sample)

			idx = -1

# generate samples from paragraphs
unlabeled_samples = get_samples_from_paras(unlabeled_paras)

with open("data/unlabeled_samples.pkl", "wb") as fp:
	pickle.dump(unlabeled_samples, fp)