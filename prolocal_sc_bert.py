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

with open("data/train_samples.pkl", "rb") as fp:
	train_samples = pickle.load(fp)

with open("data/test_samples.pkl", "rb") as fp:
	test_samples = pickle.load(fp)

with open("data/dev_samples.pkl", "rb") as fp:
	dev_samples = pickle.load(fp)

with open("configs/prolocal_sc_bert.json", "r") as fp:
	configs = json.load(fp)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)

output_path = configs["output_path"]
model_path = configs["model_path"]
patience = int(configs["patience"])
threshold = float(configs["threshold"])

seed = int(configs["seed"])

torch.manual_seed(seed)
random.seed(seed)

tokenizer = bert_tokenization.BertTokenizer.from_pretrained(pretrained_model_name_or_path = "bert-base-uncased", do_lower_case = True)
model = bert_modeling.BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path = "bert-base-uncased", num_labels = 4).cuda()

if configs["num_labeled_samples"] != "all":
	train_samples = random.sample(train_samples, int(configs["num_labeled_samples"]))

# optimizer = torch.optim.Adadelta(model.parameters(), lr = 0.2, rho = 0.95)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
	{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
	{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

max_epochs = configs["max_epochs"]

optimizer = bert_optimization.BertAdam(optimizer_grouped_parameters,
									lr = 5e-5,
									warmup = 0.1,
									t_total = max_epochs * len(train_samples))

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
											max_seq_length = 70,
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
													max_seq_length = 70,
													tokenizer = tokenizer,
													output_mode = "classification")

max_acc = 0.

max_f1 = 0.
max_f1_epoch = 0

# Train the model
for epoch in range(1, max_epochs + 1):
	if max_f1_epoch + patience < epoch and max_f1 > threshold:
		break

	random.shuffle(train_features)

	for i, train_feature in enumerate(train_features):
		# state_label_id = get_state_label_id(train_["state_label"])

		gpu_input_ids = train_feature.input_ids.cuda()
		gpu_segment_ids = train_feature.segment_ids.cuda()
		gpu_input_mask = train_feature.input_mask.cuda()
		gpu_label_id = train_feature.label_id.cuda()

		logits = model(gpu_input_ids, token_type_ids = gpu_segment_ids, attention_mask = gpu_input_mask)
		loss = nn.CrossEntropyLoss()(logits.view(-1, 4), gpu_label_id.view(-1))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i + 1) % 100 == 0:
			print("Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}".format(epoch, max_epochs, i + 1, len(train_samples), loss.item()))

	# proLocal.print_debug = True

	with torch.no_grad():
		correct_state_label = 0
		total_state_label = 0

		sum_loss = 0.0

		state_label_cm = [[0] * 4 for _ in range(4)]

		fpDev = open("dev_preds_epoch%d.txt" % (epoch), "w")

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

		torch.save(model.state_dict(), model_path + "epoch%03d.pt" % (epoch))

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