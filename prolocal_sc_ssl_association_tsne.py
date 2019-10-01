import csv
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
		sim_ab = torch.mm(labeled_embs, torch.t(unlabeled_embs))
		p_ab = torch.nn.functional.softmax(sim_ab)
		p_ab_nonzero = torch.add(p_ab, 1e-8)

		p_ab_log = torch.nn.functional.log_softmax(p_ab_nonzero)

		p_target = torch.tensor((), dtype = torch.float32)
		p_target = p_target.new_full(p_ab_nonzero.size(), 1. / float(unlabeled_embs.shape[0]))

		visit_loss = torch.mean(torch.sum(-p_target * p_ab_log, dim = 1))

		return visit_loss

	def walker_loss(self, labeled_embs, labels, unlabeled_embs):
		eq_matrix = labels.expand(labels.shape[1], labels.shape[1]).eq(labels.t().expand(labels.shape[1], labels.shape[1])).float()
		p_target = eq_matrix / torch.sum(eq_matrix, dim = 1).float()

		sim_ab = torch.mm(labeled_embs, unlabeled_embs.t())
		p_ab = torch.nn.functional.softmax(sim_ab)
		p_ba = torch.nn.functional.softmax(sim_ab.t())
		p_aba = torch.mm(p_ab, p_ba)
		p_aba_nonzero = torch.add(p_aba, 1e-8)

		p_aba_log = torch.nn.functional.log_softmax(p_aba_nonzero).float()
		walker_loss = torch.mean(torch.sum(-torch.mul(p_target, p_aba_log), dim = 1))

		return walker_loss

	def loss(self, labeled_samples, unlabeled_samples, visit_weight = 1., walker_weight = 1.):
		loss = None
		loss_empty = True

		labeled_embs = []
		unlabeled_embs = []
		labels = []

		# calculate ce loss
		for labeled_sample in labeled_samples:
			self(labeled_sample)
			labeled_embs.append(self.emb)
			labels.append(get_state_label_id(labeled_sample["state_label"])[0])

			if loss_empty:
				loss = self.ce_loss(get_state_label_id(labeled_sample["state_label"]))
				loss_empty = False
			else:
				loss += self.ce_loss(get_state_label_id(labeled_sample["state_label"]))

		for unlabeled_sample in unlabeled_samples:
			self(unlabeled_sample)
			unlabeled_embs.append(self.emb)

		labeled_embs = torch.cat(labeled_embs, dim = 0)
		unlabeled_embs = torch.cat(unlabeled_embs, dim = 0)
		labels = torch.tensor(labels).view(1, -1)

		# calculate visit loss
		loss += torch.mul(self.visit_loss(labeled_embs, unlabeled_embs), visit_weight)

		# calculate walker loss
		loss += torch.mul(self.walker_loss(labeled_embs, labels, unlabeled_embs), walker_weight)

		return loss		

with open("data/test_samples.pkl", "rb") as fp:
	test_samples = pickle.load(fp)

with open("data/unlabeled_samples.pkl", "rb") as fp:
	unlabeled_samples = pickle.load(fp)

with open("configs/prolocal_sc_ssl_association_tsne.json", "r") as fp:
	configs = json.load(fp)

model_path = configs["model_path"]

proLocal = ProLocal(emb_size = configs["emb_size"])

proLocal.load_state_dict(torch.load(model_path))

labels = []
embs = []
colour_codes = ["#00ffdf", "#c90000", "#ffe700", "#6300ff", "#c9c9c9"]

with torch.no_grad():
	for i, test_sample in enumerate(test_samples):
		_ = proLocal(test_sample)

		embs.append(proLocal.emb.detach().numpy())
		labels.append(get_state_label_id(test_sample["state_label"])[0])

	for i, unlabeled_sample in enumerate(unlabeled_samples[:200]):
		_ = proLocal(unlabeled_sample)

		embs.append(proLocal.emb.detach().numpy())
		labels.append(4)

embs = np.array(embs).reshape((-1, 20))
points_tsne = TSNE(learning_rate = 100).fit_transform(embs)
# points_tsne = PCA().fit_transform(embs)

colours = [colour_codes[label] for label in labels]

plt.scatter(points_tsne[:, 0], points_tsne[:, 1], c = colours)
plt.show()