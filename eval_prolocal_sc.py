import csv
import json
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
	def __init__(self):
		super(ProLocal, self).__init__()

		self.lstm = nn.LSTM(input_size = 102, hidden_size = 50, bidirectional = True, num_layers = 2, dropout = 0.2)		# plus verb + entity tags

		self.bilinear_agg = nn.Bilinear(100, 200, 1)		# hi * B * hev + b

		self.agg_feedforward = nn.Sequential(
			# nn.Linear(100, 4),
			nn.Dropout(p = 0.3),
			nn.Linear(100, 20),
			nn.ReLU(),
			nn.Linear(20, 4)
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

		loss_state_change_label = nn.CrossEntropyLoss()(self.state_change_label_logits, state_change_label)

		return loss_state_change_label

with open("data/test_samples.pkl", "rb") as fp:
	test_samples = pickle.load(fp)

with open("configs/eval_prolocal_sc.json", "r") as fp:
	configs = json.load(fp)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)

# state_label_weights = np.array([2.4632272228320526, 4.408644400785855, 7.764705882352941, 4.194392523364486])
# state_label_weights = np.array([math.sqrt(2.4632272228320526), math.sqrt(4.408644400785855), math.sqrt(7.764705882352941), math.sqrt(4.194392523364486)])

proLocal = ProLocal()
# proLocal.apply(weights_init)

output_path = configs["output_path"]
model_path = configs["model_path"]

proLocal.load_state_dict(torch.load(model_path))
proLocal.eval()

output_json = {}

with torch.no_grad():
	correct_state_label = 0
	total_state_label = 0

	sum_loss = 0.0

	state_label_cm = [[0] * 4 for _ in range(4)]

	fpTest = open("test_preds.txt", "w")

	for test_sample in test_samples:
		pred_state_change = proLocal(test_sample)

		state_label_id = get_state_label_id(test_sample["state_label"])

		loss = proLocal.loss(state_label_id)

		_, pred_state_label = torch.max(pred_state_change.data, 1)
		
		if pred_state_label[0] == get_state_label_id(test_sample["state_label"])[0]:
			correct_state_label += 1

		state_label_cm[int(pred_state_label[0])][get_state_label_id(test_sample["state_label"])[0]] += 1

		total_state_label += 1

		sum_loss += loss

		fpTest.write("Sentence: %s, participant: %s\n" % (test_sample["text"], test_sample["participant"]))
		fpTest.write("True state change: %s, predicted: %s\n" % (state_label_id, pred_state_change))

	fpTest.close()

	acc = 100. * float(correct_state_label) / float(total_state_label)
	f1 = cal_f1(state_label_cm)

	print("Test accuracy is: {:.3f}%, Avg loss = {:.3f}, F1 = {:.3f}"
		.format(acc, sum_loss / float(total_state_label), f1))
	print("State label confusion matrix: ", state_label_cm)
	print("\n\n=========================================================\n\n")

output_json["f1"] = f1
output_json["acc"] = acc
output_json["cm"] = state_label_cm

with open(output_path, "w") as fp:
	json.dump(output_json, fp, indent = 4)