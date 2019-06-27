import csv
import numpy as np
import pickle
import random
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

# pytorch util functions

def get_state_label_id(state_label):
	if state_label == "none":
		return np.array([0])
	elif state_label == "create":
		return np.array([1])
	elif state_label == "destroy":
		return np.array([2])
	elif state_label == "move":
		return np.array([3])

def get_seq_tags_id(state_label, span, text_len):
	# 0 = outside
	# 1 = b before loc
	# 2 = i before loc
	# 3 = b after loc
	# 4 = i after loc

	if state_label == "none":
		return np.zeros(text_len)
	elif state_label == "create":
		if span == ():
			return np.zeros(text_len)
		else:
			res = np.zeros(text_len)
			res[span[0]: (span[1] + 1)] = 4
			res[span[0]] = 3
			return res

	elif state_label == "destroy":
		if span == ():
			return np.zeros(text_len)
		else:
			res = np.zeros(text_len)
			res[span[0]: (span[1] + 1)] = 2
			res[span[0]] = 1
			return res

	elif state_label == "move":
		res = np.zeros(text_len)

		if span[0] != ():		# before span
			res[span[0][0]: (span[0][1] + 1)] = 2
			res[span[0][0]] = 1

		if span[1] != ():		# after span
			res[span[1][0]: (span[1][1] + 1)] = 4
			res[span[1][0]] = 3

		return res

# prediction model

class ProLocal(nn.Module):
	def __init__(self, state_label_weights, seq_tag_weights):
		super(ProLocal, self).__init__()

		self.state_label_weights = torch.from_numpy(state_label_weights).float()
		self.seq_tag_weights = torch.from_numpy(seq_tag_weights).float()

		self.lstm = nn.LSTM(input_size = 102, hidden_size = 50, bidirectional = True, num_layers = 2, dropout = 0.2)		# plus verb + entity tags

		self.bilinear_agg = nn.Bilinear(100, 200, 1)		# hi * B * hev + b
		self.bilinear_seq = nn.Bilinear(100, 200, 1)

		self.agg_feedforward = nn.Sequential(
			nn.Linear(100, 4)
		)

		self.seq_feedforward = nn.Sequential(
			nn.Linear(102, 5)
		)

		self.print_debug = False

	def forward(self, sample, entity):
		gloves = torch.from_numpy(sample["gloves"]).view(-1, 1, 100)
		verb_tags = torch.from_numpy(sample["verb_tags"]).view(-1, 1, 1)

		entity_tags = np.zeros(sample["verb_tags"].shape)
		entity = entity.lower()

		for idx, word in enumerate(sample["text"][:-1].split()):
			if word.lower() == entity:
				entity_tags[idx] = 1.

		entity_tags = torch.from_numpy(entity_tags).view(-1, 1, 1)

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

		# bilinear attention, sequence
		seq_attention_weights = nn.Softmax(dim = 0)(self.bilinear_seq(hidden, entity_verb_hidden.repeat(hidden.size(0), 1, 1)))
		hidden_seq = (hidden * seq_attention_weights).view(-1, 100)

		hidden_seq_aug = torch.cat((hidden.view(-1, 100), entity_tags.float().squeeze(-1), verb_tags.float().squeeze(-1)), -1)

		# classification, aggregate
		self.state_change_label_logits = self.agg_feedforward(hidden_agg).view(-1, 4)
    
		if self.print_debug:
		    print("sc logits", self.state_change_label_logits)
    
		self.state_change_label_prob = nn.Softmax(dim = 1)(self.state_change_label_logits)

		# classification, sequence
		self.seq_tags_logits = self.seq_feedforward(hidden_seq_aug).view(-1, 5)
		self.seq_tags_prob = nn.Softmax(dim = 1)(self.seq_tags_logits)

		return self.state_change_label_prob, self.seq_tags_prob

	def loss(self, state_change_label, seq_tags):
		state_change_label = torch.from_numpy(state_change_label).view(-1).long()
		seq_tags = torch.from_numpy(seq_tags).view(-1).long()

		loss_state_change_label = nn.CrossEntropyLoss(self.state_label_weights)(self.state_change_label_logits, state_change_label)

		if state_change_label[0] != 0:
			loss_seq_tags = nn.CrossEntropyLoss(self.seq_tag_weights)(self.seq_tags_prob, seq_tags)
		else:
			loss_seq_tags = 0.0

		return loss_state_change_label + loss_seq_tags

with open("data/train_samples.pkl", "rb") as fp:
	train_samples = pickle.load(fp)

with open("data/test_samples.pkl", "rb") as fp:
	test_samples = pickle.load(fp)

with open("data/dev_samples.pkl", "rb") as fp:
	dev_samples = pickle.load(fp)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)

state_label_weights = np.array([2.4632272228320526, 4.408644400785855, 7.764705882352941, 4.194392523364486])
seq_tag_weights = np.array([1.0464208242950108, 105.44262295081967, 1286.4, 37.32301740812379, 136.85106382978722])

proLocal = ProLocal(state_label_weights, seq_tag_weights)
# proLocal.apply(weights_init)

optimizer = torch.optim.Adadelta(proLocal.parameters(), lr = 0.2, rho = 0.95)
max_epoch = 20

# Train the model
for epoch in range(max_epoch):
	random.shuffle(train_samples)

	for i, train_sample in enumerate(train_samples):
		participant_idxs = list(range(len(train_sample["participants"])))
		random.shuffle(participant_idxs)

		for j in participant_idxs:
			if train_sample["text"].find(train_sample["participants"][j]) == -1:
					continue

			pred_state_change, pred_seq_tags = proLocal(train_sample, train_sample["participants"][j])

			state_label_id = get_state_label_id(train_sample["state_labels"][j])
			seq_tags_id = get_seq_tags_id(train_sample["state_labels"][j], train_sample["spans"][j], train_sample["num_words"])

			loss = proLocal.loss(state_label_id, seq_tags_id)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if (i + 1) % 100 == 0:
			print("Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}".format(epoch + 1, max_epoch, i + 1, len(train_samples), loss.item()))

	# proLocal.print_debug = True

	with torch.no_grad():
		correct_state_label = 0
		total_state_label = 0

		correct_seq_tags = 0
		total_seq_tags = 0

		sum_loss = 0.0

		state_label_cm = [[0] * 4 for _ in range(4)]
		seq_tag_cm = [[0] * 5 for _ in range(5)]

		fpDev = open("dev_preds_epoch%d.txt" % (epoch + 1), "w")

		for i, dev_sample in enumerate(dev_samples):
			participant_idxs = list(range(len(dev_sample["participants"])))
			random.shuffle(participant_idxs)

			# print(dev_sample)

			for j in participant_idxs:
				if dev_sample["text"].find(dev_sample["participants"][j]) == -1:
					continue

				pred_state_change, pred_seq_tags = proLocal(dev_sample, dev_sample["participants"][j])

				# sys.exit()

				state_label_id = get_state_label_id(dev_sample["state_labels"][j])
				seq_tags_id = get_seq_tags_id(dev_sample["state_labels"][j], dev_sample["spans"][j], dev_sample["num_words"])

				loss = proLocal.loss(state_label_id, seq_tags_id)

				_, pred_state_label = torch.max(pred_state_change.data, 1)
				
				if pred_state_label[0] == get_state_label_id(dev_sample["state_labels"][j])[0]:
					correct_state_label += 1

				state_label_cm[int(pred_state_label[0])][get_state_label_id(dev_sample["state_labels"][j])[0]] += 1

				total_state_label += 1

				if state_label_id[0] != 0:
					_, pred_seq_tag = torch.max(pred_seq_tags.data, 1)

					for k in range(seq_tags_id.shape[0]):
						seq_tag_cm[int(pred_seq_tag[k])][int(seq_tags_id[k])] += 1

						if pred_seq_tag[k] == seq_tags_id[k]:
							correct_seq_tags += 1

						total_seq_tags += 1

				sum_loss += loss

				fpDev.write("Sentence: %s, participant: %s\n" % (dev_sample["text"], dev_sample["participants"][j]))
				fpDev.write("True state change: %s, spans: %s\n" % (state_label_id, dev_sample["spans"][j]))
				fpDev.write("Predicted state change: %s\n" % pred_state_change)
				fpDev.write("Predicted sequence tags: %s\n\n" % pred_seq_tags)

		fpDev.close()

		print("Validation accuracy is: {:.3f}%, {:.3f}%, Avg loss = {:.3f}"
			.format(100 * correct_state_label / total_state_label, 100 * correct_seq_tags / total_seq_tags, sum_loss / float(total_state_label)))
		print("State label confusion matrix: ", state_label_cm)
		print("Sequence tag confusion matrix: ", seq_tag_cm)
		print("\n\n=========================================================\n\n")