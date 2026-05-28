import sys
from pathlib import Path
import streamlit as st

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from checkpoint import CheckPoint
from datasets import LetterStringDataLoader

@st.cache_resource
def get_setup(path="../../models/num_permuted_alphabets/MLC_batchalph_dallstudy1_copy_perm200_nep20.pt"):
    # Load the model checkpoint (update path as needed)
    cp = CheckPoint.from_pt(path)
    model = cp.load_model(verbose=False)
    model.eval()
    return cp, model

@st.cache_resource
def get_dataloader(path="../../data/letter-string-analogies/all_transformations_study1_copy_perm40"):
    loader = LetterStringDataLoader(
        mode="test", 
        data_dir=path, 
        batch_size=20, 
        batching_method="alphabet",
        shuffle=False
    )
    return loader
