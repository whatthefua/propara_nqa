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
	def __init__(self, emb_size = 20):
		super(ProLocal, self).__init__()

		self.emb_size = emb_size

		self.lstm = nn.LSTM(input_size = 102, hidden_size = 50, bidirectional = True, num_layers = 2, dropout = 0.2)		# plus verb + entity tags

		self.bilinear_agg = nn.Bilinear(100, 200, 1)		# hi * B * hev + b

		self.agg_feedforward_emb = nn.Sequential(
			nn.Dropout(p = 0.3),
			nn.Linear(100, emb_size),
			nn.ReLU()
		)

		self.agg_classify = nn.Linear(emb_size, 4)

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

		self.hidden_agg = (hidden * agg_attention_weights).sum(dim = 0).view(-1, 100)

		# classification, aggregate
		self.emb = self.agg_feedforward_emb(self.hidden_agg).view(-1, self.emb_size)
		self.state_change_label_logits = self.agg_classify(self.emb).view(-1, 4)
    
		if self.print_debug:
		    print("sc logits", self.state_change_label_logits)
    
		self.state_change_label_prob = nn.Softmax(dim = 1)(self.state_change_label_logits)

		return self.state_change_label_prob

	def ce_loss(self, state_change_label):
		state_change_label = torch.from_numpy(state_change_label).view(-1).long()

		loss_state_change_label = nn.CrossEntropyLoss()(self.state_change_label_logits, state_change_label)

		return loss_state_change_label

	def visit_loss(self, labeled_embs, unlabeled_embs):
		labeled_embs_torch = torch.from_numpy(labeled_embs).view(-1, self.emb_size)
		unlabeled_embs_torch = torch.from_numpy(unlabeled_embs).view(-1, self.emb_size)

		sim_ab = torch.mm(labeled_embs_torch, torch.t(unlabeled_embs_torch))
		p_ab = torch.nn.functional.softmax(sim_ab)
		p_ab_nonzero = torch.add(p_ab, 1e-8)

		p_ab_log = torch.nn.functional.log_softmax(p_ab_nonzero)

		p_target = torch.tensor((), dtype = torch.float32)
		p_target = p_target.new_full(p_ab_nonzero.size(), 1. / float(unlabeled_embs.shape[0]))

		visit_loss = torch.mean(torch.sum(-p_target * p_ab_log, dim = 1))

		return visit_loss

	def walker_loss(self, labeled_embs, labels, unlabeled_embs):
		labels_torch = torch.from_numpy(labels)
		eq_matrix = labels_torch.expand(labels.size, labels.size).eq(labels_torch.t().expand(labels.size, labels.size)).float()
		p_target = eq_matrix / torch.sum(eq_matrix, dim = 1).float()

		labeled_embs_torch = torch.from_numpy(labeled_embs).view(-1 ,self.emb_size)
		unlabeled_embs_torch = torch.from_numpy(unlabeled_embs).view(-1, self.emb_size)

		sim_ab = torch.mm(labeled_embs_torch, unlabeled_embs_torch.t())
		p_ab = torch.nn.functional.softmax(sim_ab)
		p_ba = torch.nn.functional.softmax(sim_ab.t())
		p_aba = torch.mm(p_ab, p_ba)
		p_aba_nonzero = torch.add(p_aba, 1e-8)

		p_aba_log = torch.nn.functional.log_softmax(p_aba_nonzero).float()
		walker_loss = torch.mean(torch.sum(-torch.mul(p_target, p_aba_log), dim = 1))

		return walker_loss

	def loss(self, labeled_samples, unlabeled_samples, visit_weight = 1., walker_weight = 1.):
		losses = []

		labeled_embs = []
		unlabeled_embs = []
		labels = []

		# calculate ce loss
		for labeled_sample in labeled_samples:
			self(labeled_sample)
			labeled_embs.append(self.emb.detach().numpy())
			labels.append(get_state_label_id(labeled_sample["state_label"])[0])

			losses.append(proLocal.ce_loss(get_state_label_id(labeled_sample["state_label"])))

		for unlabeled_sample in unlabeled_samples:
			self(unlabeled_sample)
			unlabeled_embs.append(self.emb.detach().numpy())

		labeled_embs = np.array(labeled_embs)
		unlabeled_embs = np.array(unlabeled_embs)
		labels = np.array(labels)

		# calculate visit loss
		losses.append(torch.mul(self.visit_loss(labeled_embs, unlabeled_embs), visit_weight))

		# calculate walker loss
		losses.append(torch.mul(self.walker_loss(labeled_embs, labels, unlabeled_embs), walker_weight))

		return sum(losses)
		

with open("data/train_samples.pkl", "rb") as fp:
	train_samples = pickle.load(fp)

with open("data/dev_samples.pkl", "rb") as fp:
	dev_samples = pickle.load(fp)

with open("data/unlabeled_samples.pkl", "rb") as fp:
	unlabeled_samples = pickle.load(fp)

with open("configs/prolocal_sc_ssl_association.json", "r") as fp:
	configs = json.load(fp)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)

max_iterations = configs["max_iterations"]
iteration_size = configs["iteration_size"]
output_path = configs["output_path"]
model_path = configs["model_path"]
patience = int(configs["patience"])
threshold = float(configs["threshold"])
seed = int(configs["seed"])

torch.manual_seed(seed)
random.seed(seed)

# state_label_weights = np.array([2.4632272228320526, 4.408644400785855, 7.764705882352941, 4.194392523364486])
# state_label_weights = np.array([math.sqrt(2.4632272228320526), math.sqrt(4.408644400785855), math.sqrt(7.764705882352941), math.sqrt(4.194392523364486)])
# state_label_weights = np.ones((4,))

proLocal = ProLocal(emb_size = configs["emb_size"])
# proLocal.apply(weights_init)

optimizer = torch.optim.Adadelta(proLocal.parameters(), lr = 0.2, rho = 0.95)

visit_weight = float(configs["visit_weight"])
walker_weight = float(configs["walker_weight"])

if configs["num_labeled_samples"] != "all":
	train_samples = random.sample(train_samples, int(configs["num_labeled_samples"]))

if configs["num_unlabeled_samples"] != "all":
	unlabeled_samples = random.sample(unlabeled_samples, int(configs["num_unlabeled_samples"]))

output_json = {
	"training_outputs": []
}

# max_acc = 0.
# max_acc_f1 = 0.
# max_acc_iteration = 0

max_f1 = 0.
max_f1_acc = 0.
max_f1_iteration = 0

# Train the model (semi-supervised)
for iteration in range(1, max_iterations + 1):
	if max_f1_iteration + patience < iteration and max_f1 > threshold:
		break

	train_samples_batch = random.sample(train_samples, iteration_size)
	unlabeled_samples_batch = random.sample(unlabeled_samples, iteration_size)
	# sum_loss = 0.

	loss = proLocal.loss(train_samples_batch, unlabeled_samples_batch, visit_weight, walker_weight)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	# 	sum_loss += loss.item()

	print("Iteration [{}/{}], Avg Loss: {:.3f}".format(iteration, max_iterations, loss.item() / float(iteration_size * 2)))

	# proLocal.print_debug = True

	if iteration % 20 == 0:
		with torch.no_grad():
			correct_state_label = 0
			total_state_label = 0

			sum_loss = 0.0

			state_label_cm = [[0] * 4 for _ in range(4)]

			# fpDev = open("dev_preds_epoch%d.txt" % (epoch + 1), "w")

			for i, dev_sample in enumerate(dev_samples):
				pred_state_change = proLocal(dev_sample)

				state_label_id = get_state_label_id(dev_sample["state_label"])

				loss = proLocal.ce_loss(state_label_id)

				_, pred_state_label = torch.max(pred_state_change.data, 1)
				
				if pred_state_label[0] == get_state_label_id(dev_sample["state_label"])[0]:
					correct_state_label += 1

				state_label_cm[int(pred_state_label[0])][get_state_label_id(dev_sample["state_label"])[0]] += 1

				total_state_label += 1

				sum_loss += loss

			# 	fpDev.write("Sentence: %s, participant: %s\n" % (dev_sample["text"], dev_sample["participant"]))
			# 	fpDev.write("True state change: %s, predicted: %s\n" % (state_label_id, pred_state_change))

			# fpDev.close()

			acc = 100. * float(correct_state_label) / float(total_state_label)
			f1 = cal_f1(state_label_cm)

			print("Validation accuracy is: {:.3f}%, Avg loss = {:.3f}, F1 = {:.3f}"
				.format(acc, sum_loss / float(total_state_label), f1))
			print("State label confusion matrix: ", state_label_cm)
			print("\n\n=========================================================\n\n")

			output_json["training_outputs"].append({
					"iteration": iteration,
					"accuracy": acc,
					"f1": f1,
					"cm": state_label_cm
				})

			# if acc > max_acc:
			# 	max_acc = acc
			# 	max_acc_iteration = iteration
			# 	max_acc_f1 = f1

			if f1 > max_f1:
				max_f1 = f1
				max_f1_iteration = iteration
				max_f1_acc = acc

		torch.save(proLocal.state_dict(), model_path + "iteration%05d.pt" % (iteration))

# output_json["max_acc"] = {
# 	"acc": max_acc,
# 	"iteration": max_acc_iteration,
# 	"f1": max_acc_f1
# }

output_json["max_f1"] = {
	"f1": max_f1,
	"iteration": max_f1_iteration,
	"acc": max_f1_acc
}

with open(output_path, "w") as fp:
	json.dump(output_json, fp, indent = 4)