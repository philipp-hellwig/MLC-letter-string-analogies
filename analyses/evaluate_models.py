import os

import torch
import pandas as pd
import numpy as np

import sys 
import analysis_utils
sys.path.append("../")

from checkpoint import CheckPoint
from datasets import LetterStringDataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_accuracy_table(checkpoints:list):
    # set up dataset to store accuracies:
    all_accs = pd.DataFrame(
        columns=["filename_model", "batching_method", "num. seen alphabets in training", "seen transform.", "new transform.", "alphabets"]
    )
    # new alphabets are constant across all checkpoints
    test_new_alph = LetterStringDataLoader(
        mode="test", 
        data_dir="../data/all_transformations_study1_new_alphabets",
        batch_size=2000
    )
    for cp in checkpoints:
        model = cp.load_model(verbose=False)

        # obtain accuracies on seen training alphabets:
        test_seen_alph = LetterStringDataLoader(
            mode="test", 
            data_dir=f"../{cp.train_config.dir_data}",
            batch_size=2000
        )
        pred = analysis_utils.predict_dataset(test_seen_alph, model, alternative_rule_errors=False)
        # exclude standard alphabet:
        pred = pred[pred["n_perm"]!= 0]
        test_acc_seen_transform = np.mean(pred[pred.distribution == "in"]["correct"])
        test_acc_new_transform = np.mean(pred[pred.distribution == "out-of"]["correct"])
        all_accs.loc[all_accs.shape[0],:] = [
            cp.train_config.filename_model, 
            cp.train_config.batching_method, 
            cp.num_perm_alphs, 
            test_acc_seen_transform, 
            test_acc_new_transform,
            "seen"
        ]

        # obtain accuracies on new alphabets:
        pred = analysis_utils.predict_dataset(test_new_alph, model, alternative_rule_errors=False)
        # exclude standard alphabet:
        pred = pred[pred["n_perm"]!= 0]
        test_acc_seen_transform = np.mean(pred[pred.distribution == "in"]["correct"])
        test_acc_new_transform = np.mean(pred[pred.distribution == "out-of"]["correct"])
        all_accs.loc[all_accs.shape[0],:] = [
            cp.train_config.filename_model, 
            cp.train_config.batching_method, 
            cp.num_perm_alphs, 
            test_acc_seen_transform, 
            test_acc_new_transform,
            "new"
        ]
    return all_accs

def main():

    # batching experiments no copy tasks, 20 permuted training alphabets:
    dir_path = "../models/batching_experiments"
    folder = os.listdir(dir_path)
    cps_perm20 = []

    for filename in folder:
        cp = CheckPoint.from_pt("/".join([dir_path, filename]))
        cp.train_config.filename_model = filename
        cp.num_perm_alphs = 20
        cps_perm20.append(cp)


    # batching experiments with copy tasks, 20 permuted training alphabets:
    dir_path = "../models/copy_batching_experiments"
    folder = os.listdir(dir_path)
    cps_copy_perm20 = []

    for filename in folder:
        cp = CheckPoint.from_pt("/".join([dir_path, filename]))
        cp.train_config.filename_model = filename
        cp.num_perm_alphs = 20
        cps_copy_perm20.append(cp)


    # batching experiments with copy tasks, 200 permuted training alphabets:
    dir_path = "../models/num_permuted_alphabets"
    folder = os.listdir(dir_path)
    cps_copy_perm200 = []

    for filename in folder:
        if "200" in filename:
            cp = CheckPoint.from_pt("/".join([dir_path,filename]))
            cp.train_config.filename_model = filename
            cp.num_perm_alphs = 200
            cps_copy_perm200.append(cp)

    tbl_copy_perm200 = get_accuracy_table(cps_copy_perm200)
    tbl_copy_perm200.to_csv("copy_perm200_accuracies.csv", index=False)

    tbl_copy_perm20 = get_accuracy_table(cps_copy_perm20)
    tbl_copy_perm20.to_csv("copy_perm20_accuracies.csv", index=False)

    tbl_perm20 = get_accuracy_table(cps_perm20)
    tbl_perm20.to_csv("perm20_accuracies.csv", index=False)