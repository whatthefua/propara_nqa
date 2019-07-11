# propara_nqa

Relevant files:

prolocal.py - Original file, contains ProLocal which predicts both sentence and word tags
prolocal_sc.py - Modified ProLocal model, only predicts sentence tags
prolocal_sc_ssl.py - SSL ProLocal model using pseudo-labeling
prolocal_sc_ssl_association - SSL ProLocal model using association loss
generate_samples_sc.py - Generates pickled train/test/dev samples from csv file
generate_samples_sc_unlabeled.py - Generates pickled unlabeled samples from csv file