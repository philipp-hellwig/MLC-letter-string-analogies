"""Evaluate all models on generalization dataset."""
import os
import re

import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import sys 
import analysis_utils
sys.path.append("../")

from checkpoint import CheckPoint
from datasets import LetterStringDataLoader
import generate_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # TODO: add experimental folders and update loading all model paths
    experiment_folders = []
    all_training_runs = []

    pattern = r'perm(\d+)_'
    dir_path = "../models/num_permuted_alphabets"
    training_runs = os.listdir(dir_path)
    all_alph_runs = []
    all_rand_runs = []
    for run in training_runs:
        if "alph" in run:
            cp = CheckPoint.from_pt("/".join([dir_path,run]))
            perm = re.search(pattern, run)
            cp.num_perm_alphs = int(perm.group(1))
            all_alph_runs.append(cp)
        else:
            cp = CheckPoint.from_pt("/".join([dir_path,run]))
            perm = re.search(pattern, run)
            cp.num_perm_alphs = int(perm.group(1))
            all_rand_runs.append(cp)

    df = pd.DataFrame(columns=[
        "filename","batching method", "num. seen alphabets in training", "seen_transform.", "new_transform", "alphabets"
    ])

    # TODO: load training data
    # run evaluation on generalization experiments
    for run in all_training_runs:
        ...

if __name__ == "__main__":
    main()