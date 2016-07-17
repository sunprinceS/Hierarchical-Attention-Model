import numpy as np
import sys
import pickle
import joblib
proto = pickle.HIGHEST_PROTOCOL
emb = np.zeros((2196017, 300), dtype='float32')
with open(sys.argv[1], 'r') as infile:
    i = 0
    for line in infile:
        line = line.strip().split()
        line = list(map(float, line))
        emb[i] = np.asarray(line)
        i += 1
joblib.dump(emb, sys.argv[2], protocol=proto)
