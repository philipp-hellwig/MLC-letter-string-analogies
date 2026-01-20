import streamlit as st
import utils

from tabs.training_hist import render_training_page
from tabs.computational_graph import render_computational_graph_page
from tabs.model_choice import render_model_choice_page

# App start
st.set_page_config(page_title="MLC Model Predictions", layout="wide")

cp, model, dl = render_model_choice_page()

tab1, tab2 = st.tabs(["Training", "Computational Graph"])
with tab1:
    render_training_page(cp)

with tab2:
    render_computational_graph_page(dl, model)