import bcolz
import csv
import nltk
import pickle
import numpy as np

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

def get_wordnet_pos(word):
	tag = nltk.pos_tag([word])[0][1][0].upper()
	tag_dict = {"J": wordnet.ADJ,
				"N": wordnet.NOUN,
				"V": wordnet.VERB,
				"R": wordnet.ADV}

	return tag_dict.get(tag, wordnet.NOUN)

def get_span(sentence, phrase):
	pos = sentence.find(phrase)

	if pos == -1:
		return ()

	if pos == 0:
		start_pos = 0
		end_pos = len(phrase.split()) - 1
	else:
		start_pos = len(sentence[:pos - 1].split())
		end_pos = start_pos + len(phrase.split()) - 1

		if start_pos >= len(sentence.split()):
			return ()

		if phrase.split()[0].lower() != sentence.split()[start_pos].lower():
			return ()

	return (start_pos, end_pos)

def get_samples_from_paras(paras):
	samples = []

	for para in paras:
		for i in range(len(para["texts_gloves"])):
			spans = []
			verb_tags = list(map(lambda x: 1. if x[1][0] == "V" else 0., nltk.pos_tag(para["texts"][i][:-1].split())))

			for participant in para["participants"]:
				entity_span = get_span(para["texts"][i], participant)
				entity_tags = np.zeros(len(para["texts"][i].split()))

				if entity_span != ():
					if entity_span[1] + 1 > len(para["texts"][i].split()):
						print(para["texts"][i], participant)

					for j in range(entity_span[0], entity_span[1] + 1):
						entity_tags[j] = 1.

					if len(para["texts_gloves"][i]) != len(verb_tags) or len(para["texts_gloves"][i]) != entity_tags.size:
						print(para["texts"][i], entity_tags, verb_tags)

					sample = {
						"gloves": np.array(para["texts_gloves"][i]),
						"text": para["texts"][i],
						"lemma_text": para["lemma_texts"][i],
						"num_words": len(para["texts"][i][:-1].split()),
						"participant": participant,
						"entity_tags": np.array(entity_tags),
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

paras = []

lemmatizer = WordNetLemmatizer()

with open("data/ProPara_StateChange_Unlabeled.csv", newline='') as fp:
	csvReader = csv.reader(fp, delimiter=',')

	for row in csvReader:
		if row[1] == "SID":
			idx = int(row[0])
			texts = []
			lemma_texts = []
			states = []
			participants = [x for x in row[3:] if x != "" and x != "\n"]
		elif row[2][:6] == "PROMPT":
			prompt = row[2][8:]
		elif row[1][:5] == "event":
			texts.append(row[2])
			lemma_texts.append([lemmatizer.lemmatize(x, get_wordnet_pos(x)).lower() for x in row[2][:-1].split()])

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

			paras.append(sample)
			idx = -1

# generate samples from paragraphs
samples = get_samples_from_paras(paras)

print(len(samples), " unlabeled samples")

with open("data/unlabeled_samples.pkl", "wb") as fp:
	pickle.dump(samples, fp)