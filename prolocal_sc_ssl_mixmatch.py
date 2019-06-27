import bcolz
import csv
import math
import numpy as np
import pickle
import random
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from nltk.corpus import wordnet

# pytorch util functions

state_label_texts = ["none", "create", "destroy", "move"]

def get_state_label_id(state_label):
	if state_label == "none":
		return np.array([0])
	elif state_label == "create":
		return np.array([1])
	elif state_label == "destroy":
		return np.array([2])
	elif state_label == "move":
		return np.array([3])

def get_glove_embedding(word):
	global glove

	if word in glove:
		return glove[word]
	elif word.lower() in glove:
		return glove[word.lower()]
	else:
		return np.zeros((100,))

def cal_f1(cm):
	cm_size = len(cm)
	f1 = 0.

	for i in range(cm_size):
		if sum([cm[j][i] for j in range(cm_size)]) != 0:
			precision = float(cm[i][i]) / float(sum([cm[j][i] for j in range(cm_size)]))
		else:
			precision = 0.

		if sum([cm[i][j] for j in range(cm_size)]) != 0:
			recall = float(cm[i][i]) / float(sum([cm[i][j] for j in range(cm_size)]))
		else:
			recall = 0.

		if precision + recall != 0.:
			f1 += 2 * precision * recall / (precision + recall) / float(cm_size)

	return f1

def augment_random(sample):
	sample["gloves"] += np.random.standard_normal(sample["gloves"].shape) * sample["gloves"] * 0.05
	return sample

def augment_random_without_entity(sample):
	sample["gloves"] += np.random.standard_normal(sample["gloves"].shape) * sample["gloves"] * 0.05 * np.reshape(1 - sample["entity_tags"], [-1, 1])
	return sample

def augment_synonym(sample):
	for i in range(len(sample["lemma_text"])):
		if sample["entity_tags"][i] == 0. and random.random < 0.15:
			synonyms = wordnet.synsets(sample["lemma_text"][i])
			synonym_words = []

			for lemma in synonyms.lemmas():
				synonym_words.append(lemma.name())

			synonym_words = list(set(synonym_words))
			random.shuffle(synonym_words)

			new_word = ""

			for synonym_word in synonym_words:
				if synonym_word.find("_") == -1:
					new_word = synonym_word
					break

			if new_word != "":
				sample["lemma_text"][i] = new_word
				sample["gloves"][i, :] = get_glove_embedding(new_word)

def mixmatch_sharpen(preds, temp):
	preds_exp = np.power(preds, 1. / temp)
	factor = np.sum(preds_exp)

	return preds_exp / factor

def create_mixmatch(model, labeled_samples, unlabeled_samples, augment_function, sharpen_temp = 0.5, augmentations = 2):
	mixmatch_batch = []

	for labeled_sample in labeled_samples:
		mixmatch_batch.append({
			"gloves": augment_function(labeled_sample)["gloves"],
			"entity_tags": labeled_sample["entity_tags"],
			"verb_tags": labeled_sample["verb_tags"],
			"target": get_state_label_id(labeled_sample["state_label"]),
			"loss": "ce",
			"coefficient": 1. / float(len(labeled_samples))
		})

	for unlabeled_sample in unlabeled_samples:
		unlabeled_samples_augment = []
		unlabeled_preds = []

		for _ in range(augmentations):
			unlabeled_sample_augment = augment_function(unlabeled_sample)
			unlabeled_preds.append(model(unlabeled_sample_augment).detach().numpy())
			unlabeled_samples_augment.append(unlabeled_sample_augment)

		unlabeled_preds_mean = np.mean(unlabeled_preds, axis = 0)
		unlabeled_preds_sharpened = mixmatch_sharpen(unlabeled_preds_mean, sharpen_temp)

		for unlabeled_sample_augment in unlabeled_samples_augment:
			mixmatch_batch.append({
				"gloves": unlabeled_sample_augment["gloves"],
				"entity_tags": unlabeled_sample_augment["entity_tags"],
				"verb_tags": unlabeled_sample_augment["verb_tags"],
				"target": unlabeled_preds_sharpened,
				"loss": "mse",
				"coefficient": 1. / float(len(unlabeled_samples) * augmentations * 4)
			})

	random.shuffle(mixmatch_batch)

	return mixmatch_batch

# prediction model

class ProLocal(nn.Module):
	def __init__(self, state_label_weights):
		super(ProLocal, self).__init__()

		self.state_label_weights = torch.from_numpy(state_label_weights).float()

		self.lstm = nn.LSTM(input_size = 102, hidden_size = 50, bidirectional = True, num_layers = 2, dropout = 0.2)		# plus verb + entity tags

		self.bilinear_agg = nn.Bilinear(100, 200, 1)		# hi * B * hev + b

		self.agg_feedforward = nn.Sequential(
			nn.Linear(100, 4)
			# nn.ReLU(),
			# nn.Linear(50, 4)
		)

		self.logsoftmax = nn.LogSoftmax()

		self.print_debug = False

	def forward(self, sample):
		gloves = torch.from_numpy(sample["gloves"]).view(-1, 1, 100)
		verb_tags = torch.from_numpy(sample["verb_tags"]).view(-1, 1, 1)
		entity_tags = torch.from_numpy(sample["entity_tags"]).view(-1, 1, 1)

		input_tensor = torch.cat((gloves, verb_tags, entity_tags), 2).float()

		# hidden word embeddings
		hidden, _ = self.lstm(input_tensor)

		verb_weights = verb_tags.float() / (verb_tags.float().sum(-1).unsqueeze(-1) + 1e-13)
		verb_hidden = (hidden * verb_weights).sum(dim = 0)

		if self.print_debug:
			print("w_verb", verb_weights)
			print("verb", verb_hidden)

		entity_weights = entity_tags.float() / (entity_tags.float().sum(-1).unsqueeze(-1) + 1e-13)
		entity_hidden = (hidden * entity_weights).sum(dim = 0)

		entity_verb_hidden = torch.cat((entity_hidden, verb_hidden), 0).float().view(-1, 1, 200)

		# bilinear attention, aggregate
		agg_attention_weights = nn.Softmax(dim = 0)(self.bilinear_agg(hidden, entity_verb_hidden.repeat(hidden.size(0), 1, 1)))

		if self.print_debug:
			print("w_agg", agg_attention_weights)
			print("hidden", hidden)
			print("hidden w_agg", hidden * agg_attention_weights)
			print("sum hidden w_agg", (hidden * agg_attention_weights).sum(dim = 0))

		hidden_agg = (hidden * agg_attention_weights).sum(dim = 0).view(-1, 100)

		# classification, aggregate
		self.state_change_label_logits = self.agg_feedforward(hidden_agg).view(-1, 4)
    
		if self.print_debug:
		    print("sc logits", self.state_change_label_logits)
    
		self.state_change_label_prob = nn.Softmax(dim = 1)(self.state_change_label_logits)

		return self.state_change_label_prob

	def ce_loss(self, state_change_label, coefficient):
		state_change_label = torch.from_numpy(state_change_label).view(-1).long()

		loss_state_change_label = nn.CrossEntropyLoss(self.state_label_weights)(self.state_change_label_logits, state_change_label)

		return loss_state_change_label * coefficient

	def mse_loss(self, target_preds, coefficient):
		target_preds = torch.from_numpy(target_preds)
		return nn.functional.mse_loss(self.state_change_label_logits, target_preds) * coefficient

vectors = bcolz.open('data/6B.100.dat')[:]
words = pickle.load(open('data/6B.100_words.pkl', 'rb'))
word2idx = pickle.load(open('data/6B.100_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}

with open("data/train_samples.pkl", "rb") as fp:
	train_samples = pickle.load(fp)

with open("data/test_samples.pkl", "rb") as fp:
	test_samples = pickle.load(fp)

with open("data/dev_samples.pkl", "rb") as fp:
	dev_samples = pickle.load(fp)

with open("data/unlabeled_samples.pkl", "rb") as fp:
	unlabeled_samples = pickle.load(fp)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)

# state_label_weights = np.array([2.4632272228320526, 4.408644400785855, 7.764705882352941, 4.194392523364486])
# state_label_weights = np.array([math.sqrt(2.4632272228320526), math.sqrt(4.408644400785855), math.sqrt(7.764705882352941), math.sqrt(4.194392523364486)])
state_label_weights = np.ones((4,))

proLocal = ProLocal(state_label_weights)
# proLocal.apply(weights_init)

optimizer = torch.optim.Adadelta(proLocal.parameters(), lr = 0.2, rho = 0.95)

max_iterations = 500
iteration_size = 32

max_epoch = 5

# Train the model (supervised)
for epoch in range(max_epoch):
	random.shuffle(train_samples)

	for i, train_sample in enumerate(train_samples):
		pred_state_change = proLocal(train_sample)

		state_label_id = get_state_label_id(train_sample["state_label"])

		loss = proLocal.ce_loss(state_label_id, 1.)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i + 1) % 500 == 0:
			print("Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}".format(epoch + 1, max_epoch, i + 1, len(train_samples), loss.item()))

	# proLocal.print_debug = True

	with torch.no_grad():
		correct_state_label = 0
		total_state_label = 0

		sum_loss = 0.0

		state_label_cm = [[0] * 4 for _ in range(4)]

		fpDev = open("dev_preds_epoch%d.txt" % (epoch + 1), "w")

		for i, dev_sample in enumerate(dev_samples):
			pred_state_change = proLocal(dev_sample)

			# sys.exit()

			state_label_id = get_state_label_id(dev_sample["state_label"])

			loss = proLocal.ce_loss(state_label_id, 1.)

			_, pred_state_label = torch.max(pred_state_change.data, 1)
			
			if pred_state_label[0] == get_state_label_id(dev_sample["state_label"])[0]:
				correct_state_label += 1

			state_label_cm[int(pred_state_label[0])][get_state_label_id(dev_sample["state_label"])[0]] += 1

			total_state_label += 1

			sum_loss += loss

			fpDev.write("Sentence: %s, participant: %s\n" % (dev_sample["text"], dev_sample["participant"]))
			fpDev.write("True state change: %s, predicted: %s\n" % (state_label_id, pred_state_change))

		fpDev.close()

		print("Validation accuracy is: {:.3f}%, Avg loss = {:.3f}, F1 = {:.3f}"
			.format(100 * correct_state_label / total_state_label, sum_loss / float(total_state_label), cal_f1(state_label_cm)))
		print("State label confusion matrix: ", state_label_cm)
		print("\n\n=========================================================\n\n")

max_acc = 0.
max_acc_f1 = 0.
max_acc_cm = []

max_f1 = 0.
max_f1_acc = 0.
max_f1_cm = []

# Train the model (semi-supervised)
for iteration in range(1, max_iterations + 1):
	train_samples_batch = random.sample(train_samples, iteration_size)
	unlabeled_samples_batch = random.sample(unlabeled_samples, iteration_size)

	mixmatch_batch = create_mixmatch(proLocal, train_samples_batch, unlabeled_samples_batch, augment_random_without_entity)

	sum_loss = 0.

	for mixmatch_sample in mixmatch_batch:
		pred_state_change = proLocal(mixmatch_sample)

		if mixmatch_sample["loss"] == "ce":
			loss = proLocal.ce_loss(mixmatch_sample["target"], mixmatch_sample["coefficient"])
		elif mixmatch_sample["loss"] == "mse":
			loss = proLocal.mse_loss(mixmatch_sample["target"], mixmatch_sample["coefficient"] * 100.)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		sum_loss += loss.item()

	# print("Iteration [{}/{}], Avg Loss: {:.3f}".format(iteration, max_iterations, sum_loss / float(len(mixmatch_batch))))

	# proLocal.print_debug = True

	if iteration % 25 == 0:
		with torch.no_grad():
			correct_state_label = 0
			total_state_label = 0

			sum_loss = 0.0

			state_label_cm = [[0] * 4 for _ in range(4)]

			# fpDev = open("dev_preds_epoch%d.txt" % (epoch + 1), "w")

			for i, dev_sample in enumerate(dev_samples):
				pred_state_change = proLocal(dev_sample)

				state_label_id = get_state_label_id(dev_sample["state_label"])

				loss = proLocal.ce_loss(state_label_id, 1.)

				_, pred_state_label = torch.max(pred_state_change.data, 1)
				
				if pred_state_label[0] == get_state_label_id(dev_sample["state_label"])[0]:
					correct_state_label += 1

				state_label_cm[int(pred_state_label[0])][get_state_label_id(dev_sample["state_label"])[0]] += 1

				total_state_label += 1

				sum_loss += loss

			# 	fpDev.write("Sentence: %s, participant: %s\n" % (dev_sample["text"], dev_sample["participant"]))
			# 	fpDev.write("True state change: %s, predicted: %s\n" % (state_label_id, pred_state_change))

			# fpDev.close()

			acc = 100 * correct_state_label / total_state_label
			f1 = cal_f1(state_label_cm)

			print("Iteration [{}/{}], Validation accuracy is: {:.3f}%, Avg loss = {:.3f}, F1 = {:.3f}"
				.format(iteration, max_iterations, acc, sum_loss / float(total_state_label), f1))
			print("State label confusion matrix: ", state_label_cm)
			print("\n\n=========================================================\n\n")

			if acc > max_acc:
				max_acc = acc
				max_acc_f1 = f1
				max_acc_cm = state_label_cm

			if f1 > max_f1:
				max_f1 = f1
				max_f1_acc = acc
				max_f1_cm = state_label_cm

print("Max Acc = %lf with F1 = %lf - " % (max_acc, max_acc_f1), max_acc_cm)
print("Max F1 = %lf with Acc = %lf - " % (max_f1, max_f1_acc), max_f1_cm)