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

import bert_modeling
import bert_optimization
import bert_tokenization
import bert_utils
import utils

def visit_loss(labeled_embs, unlabeled_embs):
	sim_ab = torch.mm(labeled_embs, torch.t(unlabeled_embs))
	p_ab = torch.nn.functional.softmax(sim_ab)
	p_ab_nonzero = torch.add(p_ab, 1e-8)

	p_ab_log = torch.nn.functional.log_softmax(p_ab_nonzero)

	p_target = torch.tensor((), dtype = torch.float32).cuda()
	p_target = p_target.new_full(p_ab_nonzero.size(), 1. / float(unlabeled_embs.shape[0])).cuda()

	visit_loss = torch.mean(torch.sum(-p_target * p_ab_log, dim = 1))

	return visit_loss

def walker_loss(labeled_embs, labels, unlabeled_embs):
	eq_matrix = labels.expand(labels.shape[1], labels.shape[1]).eq(labels.t().expand(labels.shape[1], labels.shape[1])).float().cuda()
	p_target = eq_matrix / torch.sum(eq_matrix, dim = 1).float()

	sim_ab = torch.mm(labeled_embs, unlabeled_embs.t())
	p_ab = torch.nn.functional.softmax(sim_ab).cuda()
	p_ba = torch.nn.functional.softmax(sim_ab.t()).cuda()
	p_aba = torch.mm(p_ab, p_ba)
	p_aba_nonzero = torch.add(p_aba, 1e-8)

	p_aba_log = torch.nn.functional.log_softmax(p_aba_nonzero).float()
	walker_loss = torch.mean(torch.sum(-torch.mul(p_target, p_aba_log), dim = 1))

	return walker_loss

def lba_loss(model, labeled_features, unlabeled_features, visit_weight, walker_weight):
	loss = None
	loss_empty = True

	labeled_embs = []
	unlabeled_embs = []
	labels = []

	# calculate ce loss
	for labeled_feature in labeled_features:
		gpu_input_ids = labeled_feature.input_ids.cuda()
		gpu_segment_ids = labeled_feature.segment_ids.cuda()
		gpu_input_mask = labeled_feature.input_mask.cuda()
		gpu_label_id = labeled_feature.label_id.cuda()

		logits = model(gpu_input_ids, token_type_ids = gpu_segment_ids, attention_mask = gpu_input_mask)

		labeled_embs.append(model.embedding)
		labels.append(labeled_feature.label_id[0])

		if loss_empty:
			loss = nn.CrossEntropyLoss()(logits.view(-1, 4), gpu_label_id.view(-1))
			loss_empty = False
		else:
			loss += nn.CrossEntropyLoss()(logits.view(-1, 4), gpu_label_id.view(-1))

	for unlabeled_feature in unlabeled_features:
		gpu_input_ids = unlabeled_feature.input_ids.cuda()
		gpu_segment_ids = unlabeled_feature.segment_ids.cuda()
		gpu_input_mask = unlabeled_feature.input_mask.cuda()
		gpu_label_id = unlabeled_feature.label_id.cuda()

		logits = model(gpu_input_ids, token_type_ids = gpu_segment_ids, attention_mask = gpu_input_mask)

		unlabeled_embs.append(model.embedding)

	labeled_embs = torch.cat(labeled_embs, dim = 0).cuda()
	unlabeled_embs = torch.cat(unlabeled_embs, dim = 0).cuda()
	labels = torch.tensor(labels).cuda().view(1, -1)

	# calculate visit loss
	loss += torch.mul(visit_loss(labeled_embs, unlabeled_embs), torch.tensor(visit_weight).cuda())

	# calculate walker loss
	loss += torch.mul(walker_loss(labeled_embs, labels, unlabeled_embs), torch.tensor(walker_weight).cuda())

	return loss

with open("data/train_samples.pkl", "rb") as fp:
	train_samples = pickle.load(fp)

with open("data/dev_samples.pkl", "rb") as fp:
	dev_samples = pickle.load(fp)

with open("data/unlabeled_samples.pkl", "rb") as fp:
	unlabeled_samples = pickle.load(fp)

with open("configs/prolocal_sc_bert_ssl_association.json", "r") as fp:
	configs = json.load(fp)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)

max_iterations = configs["max_iterations"]
labeled_iteration_size = int(configs["labeled_iteration_size"])
unlabeled_iteration_size = int(configs["unlabeled_iteration_size"])
output_path = configs["output_path"]
model_path = configs["model_path"]
patience = int(configs["patience"])
threshold = float(configs["threshold"])
seed = int(configs["seed"])

torch.manual_seed(seed)
random.seed(seed)

tokenizer = bert_tokenization.BertTokenizer.from_pretrained(pretrained_model_name_or_path = "bert-base-uncased", do_lower_case = True)
model = bert_modeling.BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path = "bert-base-uncased", num_labels = 4).cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
	{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
	{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = bert_optimization.BertAdam(optimizer_grouped_parameters,
									lr = 5e-5,
									warmup = 0.1,
									t_total = max_iterations)

visit_weight = float(configs["visit_weight"])
walker_weight = float(configs["walker_weight"])

if configs["num_labeled_samples"] != "all":
	train_samples = random.sample(train_samples, int(configs["num_labeled_samples"]))

if configs["num_unlabeled_samples"] != "all":
	unlabeled_samples = random.sample(unlabeled_samples, int(configs["num_unlabeled_samples"]))

output_json = {
	"training_outputs": []
}

bert_samples = []

for i, train_sample in enumerate(train_samples):
	bert_sample = bert_utils.InputExample(guid = "train-%d" % i,
							text_a = train_sample["text"],
							label = train_sample["state_label"],
							entity = train_sample["participant"],
							sequence_id = train_sample["entity_tags"].astype(int).tolist())

	bert_samples.append(bert_sample)

train_features = bert_utils.convert_examples_to_features(bert_samples, 
											label_list = ["none", "create", "destroy", "move"],
											max_seq_length = 100,
											tokenizer = tokenizer,
											output_mode = "classification")

bert_samples = []

for i, dev_sample in enumerate(dev_samples):
	bert_sample = bert_utils.InputExample(guid = "dev-%d" % i,
							text_a = dev_sample["text"],
							label = dev_sample["state_label"],
							entity = dev_sample["participant"],
							sequence_id = dev_sample["entity_tags"].astype(int).tolist())

	bert_samples.append(bert_sample)

dev_features = bert_utils.convert_examples_to_features(bert_samples,
													label_list = ["none", "create", "destroy", "move"],
													max_seq_length = 100,
													tokenizer = tokenizer,
													output_mode = "classification")

bert_samples = []

for i, unlabeled_sample in enumerate(unlabeled_samples):
	bert_sample = bert_utils.InputExample(guid = "unlabeled-%d" % i,
							text_a = unlabeled_sample["text"],
							label = "none",
							entity = unlabeled_sample["participant"],
							sequence_id = unlabeled_sample["entity_tags"].astype(int).tolist())

	bert_samples.append(bert_sample)

unlabeled_features = bert_utils.convert_examples_to_features(bert_samples, 
											label_list = ["none", "create", "destroy", "move"],
											max_seq_length = 100,
											tokenizer = tokenizer,
											output_mode = "classification")

max_acc = 0.
max_acc_f1 = 0.
max_acc_iteration = 0

max_f1 = 0.
max_f1_acc = 0.
max_f1_iteration = 0

# Train the model (semi-supervised)
for iteration in range(1, max_iterations + 1):
	if max_acc_iteration + patience < iteration and max_acc > threshold:
		break

	train_features_batch = random.sample(train_features, labeled_iteration_size)
	unlabeled_features_batch = random.sample(unlabeled_features, unlabeled_iteration_size)
	# sum_loss = 0.

	loss = lba_loss(model, train_features_batch, unlabeled_features_batch, visit_weight, walker_weight)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	print("Iteration [{}/{}], Avg Loss: {:.3f}".format(iteration, max_iterations, loss.item() / float(labeled_iteration_size + unlabeled_iteration_size)))

	if iteration % 20 == 0:
		with torch.no_grad():
			correct_state_label = 0
			total_state_label = 0

			sum_loss = 0.0

			state_label_cm = [[0] * 4 for _ in range(4)]

			fpDev = open("dev_preds_iteration%d.txt" % iteration, "w")

			for i, dev_feature in enumerate(dev_features):
				gpu_input_ids = dev_feature.input_ids.cuda()
				gpu_segment_ids = dev_feature.segment_ids.cuda()
				gpu_input_mask = dev_feature.input_mask.cuda()
				gpu_label_id = dev_feature.label_id.cuda()

				logits = model(gpu_input_ids, token_type_ids = gpu_segment_ids, attention_mask = gpu_input_mask)
				loss = nn.CrossEntropyLoss()(logits.view(-1, 4), gpu_label_id.view(-1))

				_, pred_state_label = torch.max(logits.cpu().data, 1)
				
				if pred_state_label[0] == dev_feature.label_id[0]:
					correct_state_label += 1

				state_label_cm[pred_state_label[0]][dev_feature.label_id[0]] += 1

				total_state_label += 1

				sum_loss += loss

				fpDev.write("Sentence: %s, participant: %s\n" % (dev_sample["text"], dev_sample["participant"]))
				fpDev.write("True state change: %s, predicted: %s\n" % (dev_feature.label_id[0], pred_state_label[0]))

			fpDev.close()

			acc = 100. * float(correct_state_label) / float(total_state_label)
			f1 = utils.cal_f1(state_label_cm)

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

			if acc > max_acc:
				max_acc = acc
				max_acc_iteration = iteration
				max_acc_f1 = f1

			if f1 > max_f1:
				max_f1 = f1
				max_f1_iteration = iteration
				max_f1_acc = acc

			torch.save(model.state_dict(), model_path + "iteration%03d.pt" % (iteration))

output_json["max_acc"] = {
	"acc": max_acc,
	"iteration": max_acc_iteration,
	"f1": max_acc_f1
}

output_json["max_f1"] = {
	"f1": max_f1,
	"iteration": max_f1_iteration,
	"acc": max_f1_acc
}

with open(output_path, "w") as fp:
	json.dump(output_json, fp, indent = 4)