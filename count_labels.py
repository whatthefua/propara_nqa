import pickle
import math
import numpy as np

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


with open("data/train_samples.pkl", "rb") as fp:
	train_samples = pickle.load(fp)

state_label_count = [0] * 4
state_label_values = ["none", "create", "destroy", "move"]

for train_sample in train_samples:
	for i in range(4):
		if state_label_values[i] == train_sample["state_label"]:
			state_label_count[i] += 1

			break

print(state_label_count)

print([(sum(state_label_count) / x) for x in state_label_count])