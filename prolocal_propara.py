import csv
import numpy as np
import pickle
import random
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

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
	def __init__(self, state_label_weights, seq_tag_weights):
		super(ProLocal, self).__init__()

		self.state_label_weights = torch.from_numpy(state_label_weights).float()
		self.seq_tag_weights = torch.from_numpy(seq_tag_weights).float()

		self.lstm = nn.LSTM(input_size = 102, hidden_size = 50, bidirectional = True, num_layers = 2, dropout = 0.3)		# plus verb + entity tags

		self.bilinear_agg = nn.Bilinear(100, 200, 1)		# hi * B * hev + b
		self.bilinear_seq = nn.Bilinear(100, 200, 1)

		self.agg_feedforward = nn.Sequential(
			nn.Linear(100, 4)
		)

		self.seq_feedforward = nn.Sequential(
			nn.Linear(102, 5)
		)

		self.print_debug = False

	def forward(self, sample):
		if self.print_debug:
			print(sample)

		gloves = torch.from_numpy(sample["text_gloves"]).view(-1, 1, 100).float()

		verb_tags = torch.from_numpy(sample["verb_tags"]).view(-1, 1, 1).float()
		entity_tags = torch.from_numpy(sample["entity_tags"]).view(-1, 1, 1).float()

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

optimizer = torch.optim.Adadelta(proLocal.parameters(), lr = 0.2, rho = 0.9)
max_epoch = 200

max_state_f1 = 0.
max_tag_f1 = 0.

# Train the model
for epoch in range(max_epoch):
	# proLocal.print_debug = True

	random.shuffle(train_samples)

	for i, train_sample in enumerate(train_samples):
		pred_state_change, pred_seq_tags = proLocal(train_sample)
		loss = proLocal.loss(train_sample["state_label"], train_sample["seq_label"])

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
			pred_state_change, pred_seq_tags = proLocal(dev_sample)
			loss = proLocal.loss(dev_sample["state_label"], dev_sample["seq_label"])

			_, pred_state_label = torch.max(pred_state_change.data, 1)
			
			if pred_state_label[0] == dev_sample["state_label"][0]:
				correct_state_label += 1

			state_label_cm[int(pred_state_label[0])][dev_sample["state_label"][0]] += 1

			total_state_label += 1

			if dev_sample["state_label"][0] != 0:
				_, pred_seq_tag = torch.max(pred_seq_tags.data, 1)

				for k in range(dev_sample["seq_label"].shape[0]):
					seq_tag_cm[int(pred_seq_tag[k])][int(dev_sample["seq_label"][k])] += 1

					if int(pred_seq_tag[k]) == int(dev_sample["seq_label"][k]):
						correct_seq_tags += 1

					total_seq_tags += 1

			sum_loss += loss.item()

			fpDev.write("Sentence: %s\n" % (dev_sample["text"]))
			fpDev.write("True state change: %s\n" % (dev_sample["state_label"]))
			fpDev.write("Predicted state change: %s\n" % pred_state_change)
			fpDev.write("Predicted sequence tags: %s\n\n" % pred_seq_tags)

		fpDev.close()

		state_label_f1 = cal_f1(state_label_cm)
		seq_tag_f1 = cal_f1(seq_tag_cm)

		print("Validation accuracy is: {:.3f}%, {:.3f}%, Avg loss = {:.3f}"
			.format(100 * correct_state_label / total_state_label, 100 * correct_seq_tags / total_seq_tags, sum_loss / float(total_state_label)))
		print("State label confusion matrix: ", state_label_cm, " F1: ", state_label_f1)
		print("Sequence tag confusion matrix: ", seq_tag_cm, "F1: ", seq_tag_f1)

		if state_label_f1 > max_state_f1 or seq_tag_f1 > max_tag_f1:
			max_state_f1 = max(max_state_f1, state_label_f1)
			max_tag_f1 = max(max_tag_f1, seq_tag_f1)

			torch.save(proLocal.state_dict(), "models/epoch%03d.pt" % (epoch + 1))
			print("Saved to models/epoch%03d.pt" % (epoch + 1))

		print("\n\n=========================================================\n\n")