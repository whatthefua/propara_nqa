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

with open("data/test_samples.pkl", "rb") as fp:
	test_samples = pickle.load(fp)

with open("configs/eval_prolocal_sc_bert.json", "r") as fp:
	configs = json.load(fp)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)

output_path = configs["output_path"]
model_path = configs["model_path"]

tokenizer = bert_tokenization.BertTokenizer.from_pretrained(pretrained_model_name_or_path = "bert-base-uncased", do_lower_case = True)
model = bert_modeling.BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path = "bert-base-uncased", num_labels = 4).cuda()

model.load_state_dict(torch.load(model_path))
model.eval()

bert_samples = []

for i, test_sample in enumerate(test_samples):
	bert_sample = bert_utils.InputExample(guid = "train-%d" % i,
							text_a = test_sample["text"],
							text_b = None,
							label = test_sample["state_label"],
							entity = test_sample["participant"],
							sequence_id = test_sample["entity_tags"].astype(int).tolist())

	bert_samples.append(bert_sample)

test_features = bert_utils.convert_examples_to_features(bert_samples, 
											label_list = ["none", "create", "destroy", "move"],
											max_seq_length = 70,
											tokenizer = tokenizer,
											output_mode = "classification")

with torch.no_grad():
	correct_state_label = 0
	total_state_label = 0

	sum_loss = 0.0

	state_label_cm = [[0] * 4 for _ in range(4)]

	# fpDev = open("test_preds_epoch%d.txt" % (epoch), "w")

	for i, test_feature in enumerate(test_features):
		gpu_input_ids = test_feature.input_ids.cuda()
		gpu_segment_ids = test_feature.segment_ids.cuda()
		gpu_input_mask = test_feature.input_mask.cuda()
		gpu_label_id = test_feature.label_id.cuda()

		logits = model(gpu_input_ids, token_type_ids = gpu_segment_ids, attention_mask = gpu_input_mask)
		loss = nn.CrossEntropyLoss()(logits.view(-1, 4), gpu_label_id.view(-1))

		_, pred_state_label = torch.max(logits.cpu().data, 1)
		
		if pred_state_label[0] == test_feature.label_id[0]:
			correct_state_label += 1

		state_label_cm[pred_state_label[0]][test_feature.label_id[0]] += 1

		total_state_label += 1

		sum_loss += loss

	# 	fpDev.write("Sentence: %s, participant: %s\n" % (dev_sample["text"], dev_sample["participant"]))
	# 	fpDev.write("True state change: %s, predicted: %s\n" % (dev_feature.label_id[0], pred_state_label[0]))

	# fpDev.close()

	acc = 100. * float(correct_state_label) / float(total_state_label)
	f1 = cal_f1(state_label_cm)

	print("Testing accuracy is: {:.3f}%, Avg loss = {:.3f}, F1 = {:.3f}"
		.format(acc, sum_loss / float(total_state_label), f1))
	print("State label confusion matrix: ", state_label_cm)
	print("\n\n=========================================================\n\n")

output_json = {
	"accuracy": acc,
	"f1": f1,
	"cm": state_label_cm
}

with open(output_path, "w") as fp:
	json.dump(output_json, fp, indent = 4)