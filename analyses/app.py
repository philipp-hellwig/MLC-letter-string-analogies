import torch
import os
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"
import streamlit as st

import sys
import analysis_utils
sys.path.append("../")
import checkpoint
import evaluate

st.set_page_config(page_title="MLC Model Predictions", layout="wide")
tab1, tab2, tab3 = st.tabs(["Model","Training","Predictions"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def get_setup(path="../models/MLC_all_trans_study1_nep20_lr0_0005_bs64_dr0_2_io_easy_val.pt"):

    # Load the model checkpoint (update path as needed)
    cp = checkpoint.CheckPoint(path=path, device=DEVICE)
    model = cp.load_model()
    model.eval()

    val_loader = cp.load_dataloaders("../data/all_transformations_study1", use_datasets=["val"])[0]
    batch = next(iter(val_loader))

    predictions, logits = evaluate.predict(batch, model, val_loader.dataset.langs, max_length=val_loader.dataset.yq_max+5, return_logits=True)

    probs = torch.nn.functional.softmax(logits, dim=1)
    lang = val_loader.dataset.langs["output"]

    return cp, model, val_loader, batch, predictions, probs, lang


def plot_token_predictions(id: int, batch, predictions, probs, symbols):
    """
    Create an interactive plotly visualization of the prompt and predicted tokens.
    Hovering over a predicted token will show the top5 probabilities in the distribution.
    """
    prompt_text = batch["xq_context"][id]
    pred_text = predictions[id]
    
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
        prob = probs[id, :, i].cpu().numpy()  # shape: [vocab_size]
        topk = np.argsort(prob)[::-1][:5]
        topk_tokens = [symbols[tok] for tok in topk]
        topk_probs = [prob[tok] for tok in topk]
        tooltip = "<br>".join([f"{tok}: {p:.2%}" for tok, p in zip(topk_tokens, topk_probs)])
        customdata.append(tooltip)

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

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=500,
        showlegend=False,
        font=dict(size=18)  # Increase overall font size (axes, title, etc.)
    )
    return fig

with st.sidebar:
    models_dir = "../models/"
    model_names = os.listdir(models_dir)
    model_names = [model for model in model_names if "MLC_batch" in model]
    path = models_dir + st.radio("Choose a model:", options=model_names)
    cp, model, val_loader, batch, predictions, probs, lang = get_setup(path)


with tab1:
    st.subheader("Model Comparisons")
    with open("table_batching.md", "r") as f:
        model_comparisons_table = f.read()
    st.markdown(model_comparisons_table, unsafe_allow_html=True)
    st.subheader("Model Specifications")
    st.markdown(model.__str__().replace("\n", "<br>"), unsafe_allow_html=True)


with tab2:
    train_figs = analysis_utils.training_information(cp, val_loader)
    st.pyplot(train_figs[0])
    st.pyplot(train_figs[1])


with tab3:
    id = st.number_input("Problem in batch", min_value=0, max_value=len(batch["xq_context"])-1, value=0)
    fig = plot_token_predictions(id, batch, predictions, probs, [lang.index2symbol[i] for i in range(lang.n_symbols)])

    st.plotly_chart(fig, use_container_width=True)