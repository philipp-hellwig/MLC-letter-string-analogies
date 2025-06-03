import torch
import os

import plotly.io as pio
import streamlit as st

import sys
import analysis_utils
sys.path.append("../")
from checkpoint import CheckPoint
import evaluate


pio.renderers.default = "browser"

st.set_page_config(page_title="MLC Model Predictions", layout="wide")
tab1, tab2, tab3 = st.tabs(["Model","Training","Predictions"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def get_setup(path="../models/MLC_all_trans_study1_nep20_lr0_0005_bs64_dr0_2_io_easy_val.pt"):

    # Load the model checkpoint (update path as needed)
    cp = CheckPoint.from_pt(path)
    model = cp.load_model(verbose=False)
    model.eval()

    val_loader = cp.load_dataloaders("../data/all_transformations_study1", use_datasets=["val"], verbose=False)[0]
    batch = next(iter(val_loader))

    predictions, logits = evaluate.predict(batch, model, val_loader.dataset.langs, max_length=val_loader.dataset.yq_max+5, return_logits=True)

    probs = torch.nn.functional.softmax(logits, dim=1)
    lang = val_loader.dataset.langs["output"]

    return cp, model, val_loader, batch, predictions, probs, lang


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


with tab2:
    st.header("Currently Selected Training Run")
    st.markdown(cp.__str__(), unsafe_allow_html=True)
    train_figs = analysis_utils.training_information(cp, val_loader, fig_width=10, figs_only=True)
    st.header("Training History")
    st.subheader("Loss and Learning Rate")
    st.pyplot(train_figs[0], use_container_width=False)
    st.subheader("Accuracy")
    st.pyplot(train_figs[1], use_container_width=False)
    st.subheader("By transformation type")
    st.pyplot(train_figs[2], use_container_width=False)


with tab3:
    id = st.number_input("Problem in batch", min_value=0, max_value=len(batch["xq_context"])-1, value=0)
    st.subheader(f"{batch['transformation'][id]}-problem")
    fig = analysis_utils.plot_token_predictions(id, batch, predictions, probs, [lang.index2symbol[i] for i in range(lang.n_symbols)])
    st.plotly_chart(fig)

    st.subheader("Encoder Averaged Attention Activations")
    enc_attn_fig = analysis_utils.get_encoder_attention_plot(model, batch, idx=id)
    st.pyplot(enc_attn_fig)