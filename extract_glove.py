import bcolz
import numpy as np
import pickle

words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir='data/6B.100.dat', mode='w')

with open('data/glove.6B.100d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)
    
vectors = bcolz.carray(vectors[1:].reshape((-1, 100)), rootdir='data/6B.100.dat', mode='w')
vectors.flush()
pickle.dump(words, open('data/6B.100_words.pkl', 'wb'))
pickle.dump(word2idx, open('data/6B.100_idx.pkl', 'wb'))