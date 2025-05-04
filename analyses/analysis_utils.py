import sys 

from tqdm import tqdm
import pandas as pd

sys.path.append("../")
import evaluate
import generate_data


def get_predictions_dataset(val_dataloader, model, max_length: int=20, num_batches: int=None, keep_cols: list=None) -> pd.DataFrame:
    """Convenience function that computes predictions and returns the dataset in a dataframe with added columns `pred` and `correct`"""
    data_with_pred = []
    for i, batch in enumerate(tqdm(val_dataloader, desc="Predicting")):
        batch["pred"] = evaluate.predict(batch, model, val_dataloader.dataset.langs, max_length=20)
        data_with_pred.append(batch)
        if num_batches is not None:
            if i > num_batches:
                break
    
    # combine batches into dataframe:
    # use all keys unless specified otherwise:
    if keep_cols is None:
        keep_cols = data_with_pred[0].keys()
    pred_data = pd.DataFrame(columns=keep_cols)
    for col in keep_cols:
        col_data = []
        for batch in data_with_pred:
            col_data += batch[col]
        pred_data[col] = col_data
    pred_data["correct"] = pred_data["yq"] == pred_data["pred"]
    return pred_data


def get_alternative_rule(row, check_transformations=[2,3]):
    for func_id in check_transformations:
        try:
            trans = generate_data.ALL_TRANSFORMATIONS[func_id](row.xq, row.alphabet.split())[1]
        except IndexError:
            row.alternate_rule = "na"
            return row
        if trans == row.pred:
            row.alternate_rule = generate_data.ALL_TRANSFORMATIONS[func_id].__name__
            return row
    return row