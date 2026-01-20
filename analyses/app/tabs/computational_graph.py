import streamlit as st
import random
import string
import sys
import torch
from torch.nn.functional import softmax

from d3 import d3_computational_graph

sys.path.append("../")
import analysis_utils as au
from extractor import Extractor
from decoderlens import DecoderLens
from perturber import Perturber


def prepare_prediction_data(logits, vocab_list, top_k=5):
    """
    logits: Tensor of shape [seq_len, vocab_size]
    vocab_list: List of strings e.g., ["a", "b", "c", ..., "z"]
    """
    probs = softmax(logits, dim=0) # [seq_len, 26]
    top_probs, top_indices = probs.topk(top_k, dim=0)
    seq_len = logits.size(1)
    prediction_data = []
    for i in range(seq_len):
        step_data = []
        for j in range(top_k):
            # Accessing [j, i] because topk returns [top_k, seq_len]
            step_data.append({
                "label": vocab_list[top_indices[j, i].item()],
                "prob": float(top_probs[j, i].item())
            })
        prediction_data.append(step_data)    
    return prediction_data


def render_computational_graph_page(dl, model):
    # session state initialization
    if 'deleted_points' not in st.session_state:
        st.session_state.deleted_points = set()
    
    
    # Task Selection UI

    # dataset and batch fetching:
    # setup filter (Applying logic from your snippet)
    alphabet_str = " ".join(list(string.ascii_lowercase))
    filter_options = dl.dataset.transformation_types
    # filter_options = filter_options.sort()
    filter_options = sorted(filter_options)
    
    st.header("Select Data")
    filter_by = st.selectbox("Filter dataset by transformation", filter_options)

    filter_query = [" | ".join([filter_by, alphabet_str])]
    dl.dataset.set_filter("transformation", [filter_by])

    # get batch & index
    # We use a fixed seed for reproducibility as per your snippet
    random.seed(10) 
    batch = next(iter(dl))
    
    idx = st.number_input("Problem Index in Batch", min_value=0, max_value=len(batch["xq_context"])-1, value=0)
    
    # display selected task
    st.header("Task Description")
    st.code(f'Alphabet: {batch["alphabet"][idx]}\nStudy: {batch["study"][idx]}\nProblem: {batch["problem"][idx]}')
    

    st.header("Computational Graph")

    # Perturbation Options
    st.subheader("Perturbations")
    layer = st.number_input("Predict from encoder layer:", min_value=1, max_value=3, value=3)
    layer -= 1
    reset_ablations = st.button("Reset All Ablations")
    if reset_ablations:
        st.session_state.deleted_points = set()
        st.rerun()
    
    # View options (attention averaged vs. head view)
    st.subheader("View")
    view = st.segmented_control(label="Attention View", options=["Averaged", "Individual Heads"], default="Averaged")

    # setup objects that extract and manipulate model:
    # extractor to get attention and hidden states:
    extractor = Extractor(model)
    extractor.register()
    # TODO: perturber to adjust hidden states:
    perturber = Perturber(model)
    # decoder lens to apply decoder earlier (if wanted)
    decoderlens = DecoderLens(model)

    # forward pass to extract hidden states
    with torch.no_grad():
        _ = model(batch["yq_io_padded"], batch)
    
    # apply ablations:
    modified_hidden_list = []
    for i in range(3):
        h = extractor.hidden_states[f"layer_{i}"].clone() # Clone to avoid modifying the original extractor cache
        # Apply ablation for this layer
        layer_pts = [p for p in st.session_state.deleted_points if p[0] == i]
        for _, t_idx, d_idx in layer_pts:
            if t_idx < h.size(1) and d_idx < h.size(2):
                h[:, t_idx, d_idx] = 0.0
        modified_hidden_list.append(h)
    
    # Get model predictions and logits using DecoderLens:
    pred, logits =  decoderlens.predict_batch(
        modified_hidden_list[layer],
        batch,
        dl.dataset.langs,
        20,
        return_logits=True
    )

    vocab = [dl.dataset.langs["output"].index2symbol[i] for i in range(30)]
    pred_data = prepare_prediction_data(logits[idx,:,:len(pred[idx])+1], vocab_list=vocab)

    # 4. Plotting
    # for visualizing the example, we add the EOS token and ignore the PAD tokens
    example_length = len(batch["xq_context"][idx]) + 1
    labels = au.format_context_labels(batch, idx) + ["EOS"]

    # 2. Use the D3 Component
    with st.container(border=True):
        # data to send to javascript function (nested lists for attention and hidden states)
        all_attention = []
        # initialize with the input vectors to the encoder:
        hidden_states_data = [extractor.inputs["encoder"][idx].detach().cpu().numpy().tolist()]

        for i in range(3):
            # get averaged attention matrix:
            if view == "Averaged":
                attn_matrix = extractor.attention[f"layer_{i}"][idx].mean(axis=0)
                attn_matrix = attn_matrix[:example_length, :example_length]
                attn_matrix = [attn_matrix.tolist()]
            else:
                attn_matrix = extractor.attention[f"layer_{i}"][idx]
                attn_matrix = attn_matrix[...,:example_length, :example_length]
                attn_matrix = attn_matrix.tolist()
            all_attention.append(attn_matrix)
            # Get hidden states:
            h_state = modified_hidden_list[i][idx][:len(labels), :]
            h_state_subset = h_state.detach().cpu().numpy().tolist()
            hidden_states_data.append(h_state_subset)
        
        raw_separators = au.get_delimiter_positions(batch, idx)
        clean_separators = [int(x) for x in raw_separators]

        js_deleted_list = [
            {"layer": p[0], "token": p[1], "dim": p[2]} 
            for p in st.session_state.deleted_points
        ]

        # send data to d3/frontend/main.js:
        response = d3_computational_graph(
            attention_weights=all_attention, 
            hidden_states=hidden_states_data,
            labels=labels,
            separators=clean_separators,
            prediction_data=pred_data,
            decoder_layer_index=layer,
            deleted_list=js_deleted_list 
        )

        # rerun if ablation is applied:
        if (response and response.get("type") == "ABLATE_COORDINATES"):
            new_pts = [(p['layer'], p['token'], p['dim']) for p in response["points"]]
            st.session_state.deleted_points.update(new_pts)
            st.rerun() # Immediately rerun to show new 0s and new predictions
