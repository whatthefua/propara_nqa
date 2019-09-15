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
	def __init__(self, state_label_weights):
		super(ProLocal, self).__init__()

		self.state_label_weights = torch.from_numpy(state_label_weights).float()

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

		loss_state_change_label = nn.CrossEntropyLoss(self.state_label_weights)(self.state_change_label_logits, state_change_label)

		return loss_state_change_label

with open("data/train_samples.pkl", "rb") as fp:
	train_samples = pickle.load(fp)

with open("data/test_samples.pkl", "rb") as fp:
	test_samples = pickle.load(fp)

with open("data/dev_samples.pkl", "rb") as fp:
	dev_samples = pickle.load(fp)

with open("configs/prolocal_sc.json", "r") as fp:
	configs = json.load(fp)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)

# state_label_weights = np.array([2.4632272228320526, 4.408644400785855, 7.764705882352941, 4.194392523364486])
# state_label_weights = np.array([math.sqrt(2.4632272228320526), math.sqrt(4.408644400785855), math.sqrt(7.764705882352941), math.sqrt(4.194392523364486)])
state_label_weights = np.ones((4,))

output_path = configs["output_path"]
model_path = configs["model_path"]
patience = int(configs["patience"])
threshold = float(configs["threshold"])

seed = int(configs["seed"])

torch.manual_seed(seed)
random.seed(seed)

proLocal = ProLocal(state_label_weights)
# proLocal.apply(weights_init)

if configs["num_labeled_samples"] != "all":
	train_samples = random.sample(train_samples, int(configs["num_labeled_samples"]))

optimizer = torch.optim.Adadelta(proLocal.parameters(), lr = 0.2, rho = 0.95)
max_epochs = configs["max_epochs"]

output_json = {
	"training_outputs": []
}

max_acc = 0.

max_f1 = 0.
max_f1_epoch = 0

# Train the model
for epoch in range(1, max_epochs + 1):
	if max_f1_epoch + patience < epoch and max_f1 > threshold:
		break

	random.shuffle(train_samples)

	for i, train_sample in enumerate(train_samples):
		pred_state_change = proLocal(train_sample)

		state_label_id = get_state_label_id(train_sample["state_label"])

		loss = proLocal.loss(state_label_id)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i + 1) % 500 == 0:
			print("Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}".format(epoch, max_epochs, i + 1, len(train_samples), loss.item()))

	# proLocal.print_debug = True

	with torch.no_grad():
		correct_state_label = 0
		total_state_label = 0

		sum_loss = 0.0

		state_label_cm = [[0] * 4 for _ in range(4)]

		fpDev = open("dev_preds_epoch%d.txt" % (epoch), "w")

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

		acc = 100. * float(correct_state_label) / float(total_state_label)
		f1 = cal_f1(state_label_cm)

		print("Validation accuracy is: {:.3f}%, Avg loss = {:.3f}, F1 = {:.3f}"
			.format(acc, sum_loss / float(total_state_label), f1))
		print("State label confusion matrix: ", state_label_cm)
		print("\n\n=========================================================\n\n")

		output_json["training_outputs"].append({
					"epoch": epoch,
					"accuracy": acc,
					"f1": f1,
					"cm": state_label_cm
				})

		if acc > max_acc:
			max_acc = acc
			max_acc_epoch = epoch
			max_acc_f1 = f1

		if f1 > max_f1:
			max_f1 = f1
			max_f1_epoch = epoch
			max_f1_acc = acc

		torch.save(proLocal.state_dict(), model_path + "epoch%03d.pt" % (epoch))

output_json["max_acc"] = {
	"acc": max_acc,
	"epoch": max_acc_epoch,
	"f1": max_acc_f1
}

output_json["max_f1"] = {
	"f1": max_f1,
	"epoch": max_f1_epoch,
	"acc": max_f1_acc
}

with open(output_path, "w") as fp:
	json.dump(output_json, fp, indent = 4)