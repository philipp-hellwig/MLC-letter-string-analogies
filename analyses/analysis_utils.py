import sys 

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("../")
import evaluate
import generate_data


def predict_dataset(
        val_dataloader, 
        model, 
        max_length: int=None, 
        num_batches: int=None, 
        keep_cols: list=None,
        alternative_rule_errors: bool=True
    ) -> pd.DataFrame:
    """Convenience function that computes predictions and returns the dataset in a dataframe with added columns `pred` and `correct`"""
    if max_length is None:
        max_length = val_dataloader.dataset.yq_max + 5
    data_with_pred = []
    for i, batch in enumerate(tqdm(val_dataloader, desc="Predicting")):
        batch["pred"] = evaluate.predict(batch, model, val_dataloader.dataset.langs, max_length=max_length)
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

    if alternative_rule_errors:
        pred_data["alternate_rule"] = "na"
        pred_data = pred_data.apply(get_alternative_rule, axis="columns")
    return pred_data


# TODO implement all rules (not just pred and succ rules)
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


def training_information(checkpoint, val_loader, description="Training information:"):
    print(description)
    # plot training loss and learning rate:
    fig1, ax1 = plt.subplots(1, 2, figsize=(12,4))
    ax1[0] = get_loss_plot(checkpoint, ax1[0])
    _ = ax1[0].set_title("Loss")
    ax1[1] = get_lr_plot(checkpoint, ax1[1])
    _ = ax1[1].set_title("Learning Rate")
    print(f'Best Validation Loss: {checkpoint.best_val_loss:.3f}')

    # plot accuracy overall and 
    fig2, ax2 = plt.subplots(1, 2, figsize=(11,4), width_ratios=(2,3))
    val_acc = pd.DataFrame(checkpoint.val_acc_hist)
    val_acc = pd.melt(val_acc, id_vars=['epoch'], value_vars=["in", "out-of", "overall"], var_name="distribution", value_name="accuracy")
    print(f"Best Validation Accuracy: {val_acc.loc[val_acc["distribution"] =="overall","accuracy"].max():.3f}")
    _ = sns.lineplot(val_acc[val_acc.distribution == "overall"], x="epoch", y="accuracy", color="black", ax=ax2[0])
    _ = ax2[0].set_title("Overall Accuracy")
    _ = sns.lineplot(val_acc[val_acc.distribution != "overall"], x="epoch", y="accuracy", style="distribution", ax=ax2[1])
    _ = ax2[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    _ = ax2[1].set_title("Accuracy by Distribution")

    # plot accuracies by generalization/transformation type
    val_acc = pd.DataFrame(checkpoint.val_acc_hist)
    val_acc_by_gen = pd.melt(val_acc, id_vars=['epoch'], value_vars=val_loader.dataset.transformation_types, var_name="transformation", value_name="accuracy")
    val_acc_by_gen["generalization"] = val_acc_by_gen.transformation.apply(lambda x: generate_data.STD_GENERALIZATION_TYPES[x])

    fig3, ax3 = plt.subplots(3, 1, figsize=(10,16))
    for i, gen_type in enumerate([0,2,3]):
        _ = sns.lineplot(data=val_acc_by_gen[val_acc_by_gen.generalization==gen_type], x="epoch", y="accuracy", hue="transformation", ax=ax3[i])
        _ = ax3[i].set_title(f"Validation accuracy generalization type {gen_type}")
        _ = ax3[i].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    return (fig1, fig2, fig3)


def get_lr_plot(checkpoint, ax) -> plt.Axes:
    loss_hist = pd.DataFrame(checkpoint.loss_hist)
    lr_plot = sns.lineplot(data=loss_hist, x="step", y="lr", ax=ax)
    return lr_plot


def get_loss_plot(checkpoint, ax) -> plt.Axes:
    loss_hist = pd.DataFrame(checkpoint.loss_hist)
    loss_hist = pd.melt(
        loss_hist,
        id_vars=["step"],
        value_vars=["avg_train_loss", "val_loss"],
        var_name="loss",
    )
    loss_hist["loss"] = loss_hist["loss"].str.replace("_loss", "")
    loss_plot = sns.lineplot(data=loss_hist, x="step", y="value", hue="loss", ax=ax)
    return loss_plot
