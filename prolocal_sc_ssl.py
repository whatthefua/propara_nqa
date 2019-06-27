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

	def loss(self, state_change_label):
		state_change_label = torch.from_numpy(state_change_label).view(-1).long()

		loss_state_change_label = nn.CrossEntropyLoss(self.state_label_weights)(self.state_change_label_logits, state_change_label)

		return loss_state_change_label

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
max_epoch = 20

train_samples = random.sample(train_samples, 200)

# Train the model
for epoch in range(max_epoch):
	random.shuffle(train_samples)

	for i, train_sample in enumerate(train_samples):
		pred_state_change = proLocal(train_sample)

		state_label_id = get_state_label_id(train_sample["state_label"])

		loss = proLocal.loss(state_label_id)

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

			loss = proLocal.loss(state_label_id)

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

	if epoch >= 4:
		unlabeled_samples_conf = []

		for unlabeled_sample in unlabeled_samples:
			pred_state_change = proLocal(unlabeled_sample)
			_, pred_state_label = torch.max(pred_state_change.data, 1)
			conf = torch.max(pred_state_change.data) - torch.min(pred_state_change.data)
			unlabeled_samples_conf.append((unlabeled_sample, conf, pred_state_label[0]))

		unlabeled_samples_conf.sort(key = lambda x: -x[1])

		added_samples_cnt = 0

		for unlabeled_sample_conf in unlabeled_samples_conf[:100]:
			if unlabeled_sample_conf[1] >= 0.75:
				train_sample = unlabeled_sample_conf[0]
				train_sample["state_label"] = state_label_texts[unlabeled_sample_conf[2]]
				train_samples.append(train_sample)

				unlabeled_samples.remove(unlabeled_sample_conf[0])
				added_samples_cnt += 1

		print("Added %d samples for training" % added_samples_cnt)
		print("%d unlabeled samples remaining" % len(unlabeled_samples))