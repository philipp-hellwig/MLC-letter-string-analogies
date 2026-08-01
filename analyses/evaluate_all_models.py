"""Evaluate all models trained with one-shot datasets on generalization dataset."""

import os
from pathlib import Path
import re
import sys

import analysis_utils
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.append("../")

from checkpoint import CheckPoint
from datasets import LetterStringDataLoader


def main():
    experiment_folders = ["nocopy_matched_size", "copy_batching_experiments", "num_permuted_alphabets"]
    all_training_runs = []
    all_runs_metadata = {}
    num_alph_pattern = r'perm(\d+)_'

    parent_dir = Path("../models/")

    for exp_path in experiment_folders:
        training_runs = os.listdir(parent_dir / exp_path)
        for run in training_runs:
            cp = CheckPoint.from_pt(parent_dir / exp_path / run)
            all_training_runs.append(cp)
            perm = re.search(num_alph_pattern, run)
            num_perm_alphs = int(perm.group(1))
            all_runs_metadata[run] = [
                run, # filename
                "copy" in run, # copy
                cp.train_config.batching_method, # batching method 
                num_perm_alphs, # num. seen alphabets in training
            ]

    df = pd.DataFrame(columns=[
        "filename", "copy", "batching method", "num. seen alphabets in training", "seen_transform.", "new_transform", "alphabets"
    ])

    # load new alphabets eval dataset
    test_new_alph = LetterStringDataLoader(
        mode="test", 
        data_dir="../data/all_transformations_study1_new_alphabets",
        batch_size=2000
    )

    # run evaluation on generalization experiments
    for cp in tqdm(all_training_runs, desc="Evaluating all training runs..."):
        # extract evaluation on seen alphabets:
        seen_transform_acc= round([x["in"] for x in cp.val_acc_hist][-1], 3)
        new_transform_acc = round([x["out-of"] for x in cp.val_acc_hist][-1], 3)
        df.loc[df.shape[0] + 1, :] = all_runs_metadata[cp.train_config.filename_model] + \
            [seen_transform_acc] + [new_transform_acc] + ["seen"]

        # evaluation on generalization to new alphabets:
        model = cp.load_model(verbose=False)
        model = model.to(device=DEVICE)
        pred = analysis_utils.predict_dataset(test_new_alph, model, alternative_rule_errors=False)
        # exclude standard alphabet:
        pred = pred[pred["n_perm"]!= 0]
        test_seen_transform_acc = np.mean(pred[pred.distribution == "in"]["correct"])
        test_new_transform_acc = np.mean(pred[pred.distribution == "out-of"]["correct"])
        df.loc[df.shape[0] + 1, :] = all_runs_metadata[cp.train_config.filename_model] + \
            [test_seen_transform_acc] + [test_new_transform_acc] + ["new"]

    df.to_csv("tables/all_evaluations.csv", index=False)
    print("Done. Evaluated all models and saved to {}")


if __name__ == "__main__":
    main()