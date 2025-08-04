from copy import deepcopy
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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.style.use("./figures_stylesheet.mplstyle")

def predict_dataset(
        dataloader, 
        model, 
        max_length: int=None, 
        num_batches: int=None, 
        keep_cols: list=None,
        alternative_rule_errors: bool=True,
        verbose: bool=True
    ) -> pd.DataFrame:
    """Convenience function that computes predictions and returns the dataset in a dataframe with added columns `pred` and `correct`"""
    if max_length is None:
        max_length = dataloader.dataset.yq_max + 5
    data_with_pred = []
    for i, batch in enumerate(tqdm(dataloader, desc="Generating predictions", disable=not verbose)):
        # add predictions
        batch["pred"] = predict_batch(batch, model, dataloader.dataset.langs, max_length=max_length)
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
        pred_data["applied_transformation"] = "none/other"
        pred_data = pred_data.apply(get_applied_transformation, axis="columns")
    return pred_data


def get_applied_transformation(row, check_transformations=[2,3]):
    for trans_id in check_transformations:
        try:
            trans = generate_data.ALL_TRANSFORMATIONS[trans_id]["function"](row["xq"].split(), row["alphabet"].split())[1]
            if trans == row["pred"]:
                if row["transformation"] == generate_data.ALL_TRANSFORMATIONS[trans_id]["transformation"]:
                    row["applied_transformation"] = f'{generate_data.ALL_TRANSFORMATIONS[trans_id]["transformation"]} (correct)'
                else:
                    row["applied_transformation"] = generate_data.ALL_TRANSFORMATIONS[trans_id]["transformation"]
                return row
        except IndexError:
            pass
    return row


def training_history(checkpoint, val_loader, description="Training information:", fig_width: int=10, figs_only=False):
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

    # plot accuracy overall and in- / out-of-distribution
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

    generalization_descriptions = {
        0: "Training Transformations",
        2: "New (Compositional) Transformations",
        3: "Novel Transformations"
    }
    fig3, ax3 = plt.subplots(3, 1, figsize=(fig_width,16))
    for i, gen_type in enumerate([0,2,3]):
        _ = sns.lineplot(data=val_acc_by_gen[val_acc_by_gen.generalization==gen_type], x="epoch", y="accuracy", hue="transformation", ax=ax3[i])
        _ = ax3[i].set_title(generalization_descriptions[gen_type])
        _ = ax3[i].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    fig3.tight_layout(h_pad=2)
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


def get_encoder_attention_plot(model, batch, idx: int, titles=True, enc_layer: int=None):
    batch = deepcopy(batch)
    # setup figure
    fig, ax = plt.subplots(len(model.transformer.encoder.layers), 1, figsize=(8, 20))
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
        if isinstance(idx, list):
            # for multiple tasks, change labels and average attention weights:
            max_length = np.max([len(batch["xq_context"][index]) for index in idx])
            multi_idx_ticks = []
            for index, token in enumerate(batch["xq_context"][idx[0]]):
                if token == "IO":
                    io_pos = index
                else:
                    multi_idx_ticks.append("")
            len_study = len(batch["study"][idx[0]].split())
            multi_idx_ticks[io_pos - int(len_study/4)-1] = "input"
            multi_idx_ticks[io_pos + int(len_study/4)+1] = "output"
            weights = attn_weights[idx, :max_length, :max_length].detach().cpu().numpy()  # shape: [seq_len, seq_len]
            weights = np.mean(weights, axis=0)
        else:
            example_length = len(batch["xq_context"][idx])
            weights = attn_weights[idx, :example_length, :example_length].detach().cpu().numpy()  # shape: [seq_len, seq_len]
            for index, token in enumerate(batch["xq_context"][idx]):
                if token == "SOS":
                    batch["xq_context"][idx][index] = "|"
                elif token == "IO":
                    batch["xq_context"][idx][index] = "→"
                    io_pos = index
            len_alph = len(batch["alphabet"][idx].split())
            len_study = len(batch["study"][idx].split())
            dashed_separators = [
                len_alph, 
                len_alph + 1,
                io_pos,
                io_pos + 1,
                len_alph + len_study + 1,
                len_alph + len_study + 2
            ]
            ax[i].vlines(dashed_separators, 0, weights.shape[0], color="orange", linestyles="dashed", linewidth=1)
            ax[i].hlines(dashed_separators, 0, weights.shape[0], color="orange", linestyles="dashed", linewidth=1)
        sns.heatmap(
            weights, 
            cmap="viridis", 
            xticklabels=multi_idx_ticks if isinstance(idx, list) else batch["xq_context"][idx], 
            yticklabels=multi_idx_ticks if isinstance(idx, list) else batch["xq_context"][idx], 
            ax=ax[i])
        
        if titles:
            ax[i].set_title(f"Averaged Attention, Encoder-Layer {i+1}")
        ax[i].tick_params(axis='both', which='major', labelsize=8)
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=0, ha='center', fontsize=12)
        ax[i].set_yticklabels(ax[i].get_yticklabels(), rotation=-90, va='center', fontsize=12)
        ax[i].set_aspect('equal')
        ax[i].set_xlabel("K", loc='right', labelpad=-4 if isinstance(idx, list) else 4)
        ax[i].set_ylabel("Q", loc='top', labelpad=-4 if isinstance(idx, list) else 4)

        if isinstance(idx, list):
            ax[i].tick_params('both', length=0)
            len_alph = len(batch["alphabet"][idx[0]].split())
            len_study = len(batch["study"][idx[0]].split())
            dashed_separators = [
                len_alph, 
                len_alph + 1,
                io_pos,
                io_pos + 1,
                len_alph + len_study + 1,
                len_alph + len_study + 2
            ]
            ax[i].vlines(dashed_separators, 0, weights.shape[0], color="orange", linestyles="dashed", linewidth=1)
            ax[i].hlines(dashed_separators, 0, weights.shape[0], color="orange", linestyles="dashed", linewidth=1)
            
            secondary_ticks = [
                len_alph / 2, 
                len_alph + len_study / 2 + 1, 
                len_alph + len_study + 6
            ]
            # x axis
            sec = ax[i].secondary_xaxis(location=0)
            sec.set_xticks(secondary_ticks, labels=['\nalphabet', '\nstudy', '\nquery'])
            sec.tick_params('x', length=0)
            # lines between the classes:
            sec2 = ax[i].secondary_xaxis(location=0)
            sec2.set_xticks([len_alph, len_alph + len_study + 2], labels=[])
            sec2.tick_params('x', length=35, width=1)

            # y axis:
            sec = ax[i].secondary_yaxis(location=0)
            sec.set_yticks(secondary_ticks, labels=['\nalphabet', '\nstudy', '\nquery'], rotation=-90)
            sec.tick_params('y', length=0)
            # lines between the classes:
            sec2 = ax[i].secondary_yaxis(location=0)
            sec2.set_yticks([len_alph, len_alph + len_study + 2], labels=[])
            sec2.tick_params('y', length=35, width=1)
        elif i==0:
            len_alph = len(batch["alphabet"][idx].split())
            len_study = len(batch["study"][idx].split())
            secondary_ticks = [
                len_alph / 2, 
                len_alph + len_study / 2 + 1, 
                len_alph + len_study + 5
            ]
            # x axis
            sec = ax[i].secondary_xaxis(location="top")
            sec.set_xticks(secondary_ticks, labels=['\nalphabet', '\nstudy', '\nquery'], fontsize=14)
            sec.tick_params('x', length=0)
            # lines between the classes:
            sec2 = ax[i].secondary_xaxis(location="top")
            sec2.set_xticks([len_alph, len_alph + len_study + 2], labels=[])
            sec2.tick_params('x', length=20, width=1)

    return fig

def get_encoder_study_attention_plot(model, batch, idx: int, enc_layer: int=2, titles=True, axis=None, cbar=False, max_attn_value=0.4):
    batch = deepcopy(batch)
    # load plotting defaults:
    plt.style.use("./figures_stylesheet.mplstyle")
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
    x = src
    for j in range(enc_layer-1):
        x = model.transformer.encoder.layers[j](
            x,
            src_key_padding_mask=src_key_padding_mask)
    layer = model.transformer.encoder.layers[enc_layer-1]
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
    max_length = np.max([len(batch["xq_context"][index]) for index in idx])
    multi_idx_ticks = []
    for index, token in enumerate(batch["xq_context"][idx[0]]):
        if token == "IO":
            io_pos = index
        else:
            multi_idx_ticks.append("")
    len_study = len(batch["study"][idx[0]].split())

    weights = attn_weights[idx, :max_length, :max_length].detach().cpu().numpy()  # shape: [seq_len, seq_len]
    weights = np.mean(weights, axis=0)
    study_start = len(batch["alphabet"][idx[0]].split())
    study_end = len(batch["alphabet"][idx[0]].split()) + len_study + 2
    weights = weights[study_start:study_end, study_start:study_end]

    len_alph = len(batch["alphabet"][idx[0]].split())
    len_study = len(batch["study"][idx[0]].split())
    labels = ["" for _ in range(study_end - study_start)]
    labels[io_pos - len_alph] = "→"
    labels[(io_pos - len_alph) - len_study // 4 - 1] = "input"
    labels[(io_pos - len_alph) + len_study // 4 + 1] = "output"
    _ = sns.heatmap(
            weights, 
            cmap="viridis", 
            xticklabels=labels, 
            yticklabels=labels, 
            ax=axis,
            vmin=0, vmax=max_attn_value,
            cbar=cbar)
    axis.set_xticklabels(axis.get_xticklabels(), rotation=0, ha='center', fontsize=18)
    axis.set_yticklabels(axis.get_yticklabels(), rotation=-90, va='center', fontsize=18)
    axis.set_aspect('equal')
    axis.set_xlabel("K", loc='right', labelpad=-4, fontsize=20)
    axis.set_ylabel("Q", loc='top', labelpad=-4, fontsize=20)
    # if isinstance(idx, list):
    axis.tick_params('both', length=0)
    dashed_separators = [
        0, 
        1,
        io_pos - len_alph,
        io_pos - len_alph + 1,
        len_study + 1,
        len_study + 2
    ]
    axis.vlines(dashed_separators, 0, weights.shape[0], color="orange", linestyles="dashed", linewidth=1.5)
    axis.hlines(dashed_separators, 0, weights.shape[0], color="orange", linestyles="dashed", linewidth=1.5)

    return None

def get_encoder_representations(batch, model):
    src, src_key_padding_mask = model.prep_encode(batch['xq_context_padded'])
    # get encoder representations with shape (batch size, max context length batch, repr_dimension)
    enc_representations = model.transformer.encoder.forward(
        src=src,
        src_key_padding_mask=src_key_padding_mask
    )
    # reduce dimensions to shape (batch size, repr dimension):
    enc_representations = enc_representations.mean(dim=1)
    return enc_representations 



def get_cosine_similarity_matrix(tensors: list):
    """Concatenates a list of tensors along the first dimension and takes the cosine similarities for
    each element"""
    combined = torch.cat(tensors, dim=0)
    repr_norm = F.normalize(combined, p=2, dim=1)
    return repr_norm @ repr_norm.T


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
