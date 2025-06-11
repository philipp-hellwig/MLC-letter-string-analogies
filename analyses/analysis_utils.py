import argparse
import sys 

from tqdm import tqdm
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

sys.path.append("../")
from evaluate import predict_batch
import generate_data
from checkpoint import CheckPoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        batch["pred"] = predict_batch(batch, model, val_dataloader.dataset.langs, max_length=max_length)
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


def training_information(checkpoint, val_loader, description="Training information:", fig_width: int=12, figs_only=False):
    if not figs_only:
        print(description)
    # plot training loss and learning rate:
    fig1, ax1 = plt.subplots(1, 2, figsize=(fig_width,4))
    ax1[0] = get_loss_plot(checkpoint, ax1[0])
    _ = ax1[0].set_title("Loss")
    ax1[1] = get_lr_plot(checkpoint, ax1[1])
    _ = ax1[1].set_title("Learning Rate")
    if not figs_only:
        print(f'Best Validation Loss: {checkpoint.best_val_loss:.3f}')

    # plot accuracy overall and 
    fig2, ax2 = plt.subplots(1, 2, figsize=(fig_width,4), width_ratios=(2,3))
    val_acc = pd.DataFrame(checkpoint.val_acc_hist)
    val_acc = pd.melt(val_acc, id_vars=['epoch'], value_vars=["in", "out-of", "overall"], var_name="distribution", value_name="accuracy")
    if not figs_only:
        print(f"Best Validation Accuracy: {val_acc.loc[val_acc["distribution"] =="overall","accuracy"].max():.3f}")
    _ = sns.lineplot(val_acc[val_acc.distribution == "overall"], x="epoch", y="accuracy", color="black", ax=ax2[0])
    _ = ax2[0].set_title("Overall Accuracy")
    _ = sns.lineplot(val_acc[val_acc.distribution != "overall"], x="epoch", y="accuracy", style="distribution", ax=ax2[1])
    _ = ax2[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    _ = ax2[1].set_title("Accuracy by Distribution")

    # plot accuracies by generalization/transformation type
    val_acc = pd.DataFrame(checkpoint.val_acc_hist)
    val_acc_by_gen = pd.melt(val_acc, id_vars=['epoch'], value_vars=val_loader.dataset.transformation_types, var_name="transformation", value_name="accuracy")
    trans2idx = {trans["transformation"]: idx for idx, trans in generate_data.ALL_TRANSFORMATIONS.items()}
    val_acc_by_gen["generalization"] = val_acc_by_gen.transformation.apply(
            lambda x: generate_data.ALL_TRANSFORMATIONS[trans2idx[str(x)]]["generalization_type"]
    )

    fig3, ax3 = plt.subplots(3, 1, figsize=(fig_width,16))
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


def get_encoder_attention_plot(model, batch, idx: int, titles=True):
    # setup figure
    fig, ax = plt.subplots(len(model.transformer.encoder.layers), 1, figsize=(6, 16))
    num_enc_layers = len(model.transformer.encoder.layers)
    model.eval()
    # get source mask (i.e., mask padded elements in the batch):
    src, src_key_padding_mask = model.prep_encode(batch['xq_context_padded'])
    src_mask = None
    src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )
    src_mask = F._canonical_mask(
        mask=src_mask,
        mask_name="src_mask",
        other_type=None,
        other_name="",
        target_type=src.dtype,
        check_other=False,
    )
    for i, layer in enumerate(model.transformer.encoder.layers):
        x = src
        # pass input through previous encoder layers
        for j in range(0, i):
            x = model.transformer.encoder.layers[j](
                x,
                src_key_padding_mask=src_key_padding_mask)
        # get averaged attention matrix from current layer
        _, attn_weights = layer.self_attn(
            x,
            x,
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            is_causal=False
        )
        example_length = len(batch["xq_context"][idx])
        # Select the first batch and first head
        weights = attn_weights[idx, :example_length, :example_length].detach().cpu().numpy()  # shape: [seq_len, seq_len]

        _ = sns.heatmap(weights, cmap="viridis", xticklabels=batch["xq_context"][idx], yticklabels=batch["xq_context"][idx], ax=ax[num_enc_layers-1-i])
        if titles:
            _ = ax[num_enc_layers-1-i].set_title(f"Averaged Attention, Encoder-Layer {num_enc_layers-i}")
        _ = ax[num_enc_layers-1-i].tick_params(axis='both', which='major', labelsize=8)
        _ = ax[num_enc_layers-1-i].set_xlabel("key")
        _ = ax[num_enc_layers-1-i].set_ylabel("query")
    return fig, ax


def plot_token_predictions(idx: int, batch, predictions, probs, symbols):
    """
    Create an interactive plotly visualization of the prompt and predicted tokens.
    Hovering over a predicted token will show the top5 probabilities in the distribution.
    """
    prompt_text = batch["xq_context"][idx]
    pred_text = predictions[idx]
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        row_heights=[0.5, 0.5],
                        vertical_spacing=0.05,
                        subplot_titles=("Prompt", "Prediction"))

    # Plot prompt tokens
    fig.add_trace(go.Scatter(
        x=list(range(len(prompt_text))),
        y=[1]*len(prompt_text),
        mode="text",
        text=[t.replace("SOS","|").replace("IO","->") for t in prompt_text],
        textposition="middle center",
        hoverinfo="text",
        name="Prompt"
    ), row=1, col=1)

    # Plot predicted tokens with customdata for hover
    customdata = []
    # probs shape: [batch_size, num_tokens, vocab_size]
    for i in range(probs.shape[2]):  # num_tokens (prediction sequence length)
        prob = probs[idx, :, i].cpu().numpy()  # shape: [vocab_size]
        topk = np.argsort(prob)[::-1][:5]
        topk_tokens = [symbols[tok] for tok in topk]
        topk_probs = [prob[tok] for tok in topk]
        tooltip = "<br>".join([f"{tok}: {p:.2%}" for tok, p in zip(topk_tokens, topk_probs)])
        customdata.append(tooltip)

    outcome = "correct" if pred_text == batch['yq'][idx] else "incorrect"
    fig.add_trace(go.Scatter(
        x=list(range(len(pred_text))),
        y=[1]*len(pred_text),
        mode="text",
        text=pred_text,
        textposition="bottom center",
        hovertemplate="Predicted: %{text}<br>Top 5:<br>%{customdata}<extra></extra>",
        customdata=customdata,
        name="Prediction"
    ), row=2, col=1)

    fig.add_annotation(
    text=f"Prediction outcome: {outcome}, correct answer: {' '.join(batch["yq"][idx])}",
    xref="paper", yref="paper",
    x=0.5, y=0, showarrow=False,
    font=dict(size=16)
    )
    
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=500,
        showlegend=False,
        font=dict(size=18)  # Increase overall font size (axes, title, etc.)
    )
    return fig


# TODO: finish defining run analyses
def run_analyses(include_analyses: list, checkpoint_paths: list):
    """Runs analyses that require predictions from the model (hence more efficient to run on GPU)"""

    for cp_path in checkpoint_paths:
        # load checkpoint:
        cp = CheckPoint.from_pt(cp_path)
        # load model and dataset:
        val = cp.load_dataloaders(f"../{cp.train_config.dir_data}", val_batch_size=5000, use_datasets=["val", "test"])
        model = cp.load_model()
        model.eval()
        # perform specified analyses:
        predictions = predict_dataset(val, model)
        # save results:




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include_analyses', type=str, default='test.pt', help='*REQUIRED* Filename for saving model checkpoints. Ends in .pt')
    parser.add_argument('--checkpoint_paths', type=str, default='test.pt', help='*REQUIRED* Filename for saving model checkpoints. Ends in .pt')
    args = parser.parse_args()
    run_analyses(args.include_analyses, args.checkpoint_paths)


if __name__ == "__main__":
    main()