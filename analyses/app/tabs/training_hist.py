import sys
import streamlit as st
sys.path.append("../")
import analysis_utils as au

def render_training_page(cp):
    train_figs = au.training_history(cp, 1, fig_width=10, figs_only=True)
    st.header("Training History")
    st.subheader("Val. Accuracy")
    st.pyplot(train_figs[1], use_container_width=False)
    st.subheader("Val. Accuracy By Transformation")
    st.pyplot(train_figs[2], use_container_width=False)

    st.header("Parameters")
    st.markdown(cp.__str__(), unsafe_allow_html=True)