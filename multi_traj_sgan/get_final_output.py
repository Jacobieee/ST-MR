import os
import numpy as np
import pickle

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--out_file", default="sgan_output_vp.traj.p", type=str)



if __name__ == "__main__":
    args = parser.parse_args()
    with open('files.npy', 'rb') as f:
        files = np.load(f)

    with open('sgan_out.npy', 'rb') as f:
        predictions = np.load(f)

    output = {}
    for file, pred in zip(files, predictions):
        output[file] = pred

    with open(args.out_file, "wb") as f:
        pickle.dump(output, f)
