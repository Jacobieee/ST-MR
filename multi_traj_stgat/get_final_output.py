import os
import numpy as np
import pickle

with open('files.npy', 'rb') as f:
    files = np.load(f)

with open('stgat_out.npy', 'rb') as f:
    predictions = np.load(f)

output = {}
for file, pred in zip(files, predictions):
    output[file] = pred

with open('stgat_output.traj.p', "wb") as f:
    pickle.dump(output, f)