import string
import os

import plotly.io as pio
import streamlit as st
import torch

import sys
import analysis_utils
sys.path.append("../")
from checkpoint import CheckPoint
import evaluate
from datasets import Lang, LetterStringDataset

pio.renderers.default = "browser"

st.set_page_config(page_title="MLC Model Predictions", layout="wide")
tab1, tab2, tab3, tab4 = st.tabs(["Model","Training","Predictions", "Create your own task"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_setup(path="../models/MLC_all_trans_study1_nep20_lr0_0005_bs64_dr0_2_io_easy_val.pt"):
    # Load the model checkpoint (update path as needed)
    cp = CheckPoint.from_pt(path)
    model = cp.load_model(verbose=False)
    model.eval()
    val_loader = cp.load_dataloaders(f"../{cp.train_config.dir_data}", use_datasets=["val"], verbose=False)[0]
    return cp, model, val_loader


@st.cache_data
def get_predictions_for_filter(model_path, filter_by=None):
    _, _model, _val_loader = get_setup(model_path)
    if filter_by is None:
        _val_loader.dataset.set_filter("unstructured")
    else:
        _val_loader.dataset.set_filter("transformation", [filter_by])
    batch = next(iter(_val_loader))
    predictions, logits = evaluate.predict_batch(
        batch, _model, _val_loader.dataset.langs,
        max_length=_val_loader.dataset.yq_max+5, return_logits=True
    )
    probs = torch.nn.functional.softmax(logits, dim=1)
    lang = _val_loader.dataset.langs["output"]
    return batch, predictions, probs, lang


with st.sidebar:
    experiments = os.listdir("../models")
    experiment_path = st.radio("Choose an experiment:", options=experiments)
    models_dir = f"../models/{experiment_path}/"
    model_names = os.listdir(models_dir)
    path = models_dir + st.radio("Choose a model:", options=model_names)
    cp, model, val_loader = get_setup(path)


with tab1:
    st.subheader("Model Comparisons")
    with open("results_tables/table_batching.md", "r") as f:
        model_comparisons_table = f.read()
    st.markdown(model_comparisons_table, unsafe_allow_html=True)


with tab2:
    st.header("Currently Selected Training Run")
    st.markdown(cp.__str__(), unsafe_allow_html=True)
    train_figs = analysis_utils.training_history(cp, val_loader, fig_width=10, figs_only=True)
    st.header("Training History")
    st.subheader("Loss and Learning Rate")
    st.pyplot(train_figs[0], use_container_width=False)
    st.subheader("Accuracy")
    st.pyplot(train_figs[1], use_container_width=False)
    st.subheader("By transformation type")
    st.pyplot(train_figs[2], use_container_width=False)


with tab3:
    filter_by = st.selectbox("Filter dataset by", [None] + val_loader.dataset.transformation_types)
    batch, predictions, probs, lang = get_predictions_for_filter(path, filter_by)
    idx = st.number_input("Problem in batch", min_value=0, max_value=len(batch["xq_context"])-1, value=0)
    st.subheader(f"{batch['transformation'][idx]}-problem")
    fig = analysis_utils.plot_token_predictions(idx, batch, predictions, probs, [lang.index2symbol[i] for i in range(lang.n_symbols)])
    st.plotly_chart(fig)
    st.subheader("Encoder Averaged Attention Activations")
    enc_attn_fig = analysis_utils.get_encoder_attention_plot(model, batch, idx=idx, titles=False)
    st.pyplot(enc_attn_fig, use_container_width=False)


def batch_single_task(alph, study_in, study_out, query, solution):
    alph = alph.strip()
    study_in = study_in.strip()
    study_out = study_out.strip()
    query = query.strip()
    solution = solution.strip()
    
    xq_context = (alph + " SOS " + study_in + " IO " + study_out + " SOS " + query).split(" ")

    print(xq_context)
    # Create a dummy dataset to access collate_fn:
    ds = LetterStringDataset(
        mode="val", 
        data_dir="../data/dummy",
        batch_size=1
    )
    problem = dict()
    problem["xq_context"] = xq_context
    problem["xq_context_tensor"] = ds.langs["input"].symbols_to_tensor(xq_context)
    problem["xq_context_padded"] = ds.langs["input"].symbols_to_tensor(xq_context).unsqueeze(0)
    problem["yq"] = solution.split()
    problem["yq_tensor"] = ds.langs["output"].symbols_to_tensor(problem["yq"])
    # yq shifted right (starting with io token)
    io = ds.langs["output"].index2symbol[ds.langs["output"].IN_OUT_idx]
    problem["yq_io_tensor"] =  ds.langs["output"].symbols_to_tensor([io] + problem["yq"], add_eos=False)
    return ds.collate_fn([problem]), ds


with tab4:
    with st.form("Your task:"):
        alph = st.text_input(label="Input alphabet", value=" ".join(list(string.ascii_lowercase)))
        study_in = st.text_input(label="Study Example (in)", value="a b c")
        study_out = st.text_input(label="Study Example (out)", value="a b d")
        query = st.text_input(label="Query", value="i j")
        solution = st.text_input(label="Solution", value="i k")
        submitted = st.form_submit_button("Submit")
        if submitted:
            batch, ds = batch_single_task(alph, study_in, study_out, query, solution)
            pred = evaluate.predict_batch(
                batch, model, ds.langs, 
                max_length=len(query.split(" ")*2), check_for_valid_length=False)
            st.subheader("Model Prediction:")
            st.text(" ".join(pred[0]))
