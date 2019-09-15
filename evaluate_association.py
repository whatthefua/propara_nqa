import os
import subprocess

json_str = """{
	"max_iterations": 1000,
	"iteration_size": %d,
	"num_labeled_samples": "all",
	"num_unlabeled_samples": "all",
	"emb_size": %d,
	"visit_weight": %f,
	"walker_weight": %f,
	"output_path": "%s",
	"model_path": "%s"
}"""

# for iteration_size in [64]:
# 	for emb_size in [30, 40]:

iteration_size = 128
emb_size = 50

for walker_weight in [1., 100.]:
	for visit_weight in [1., 100.]:
		model_path = "models/association_%d_%d_all_all_%.0f_%.0f/" % (iteration_size, emb_size, walker_weight, visit_weight)
		os.mkdir(model_path)

		fp = open("configs/prolocal_sc_ssl_association.json", "w")
		fp.write(json_str % (iteration_size, emb_size, visit_weight, walker_weight, "outputs/association_%d_%d_all_all_%.0f_%.0f.json" % (iteration_size, emb_size, walker_weight, visit_weight), model_path))
		fp.close()

		cmd = ["python35", "prolocal_sc_ssl_association.py"]
		subprocess.Popen(cmd).wait()