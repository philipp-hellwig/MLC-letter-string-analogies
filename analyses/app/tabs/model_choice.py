import pandas as pd
import streamlit as st
import utils


def load_performance_results():
    df = pd.read_csv("performance_table.csv")

    # 1. Identify columns that will serve as the row index (identifiers)
    id_cols = ['filename_model', 'Model', 'batching_method', 'num. seen alphabets in training']
    # 2. Set index and unstack the 'alphabets' column to create the second column level
    df_pivot = df.set_index(id_cols + ['alphabets']).unstack('alphabets')

    # 3. Define the mapping for renaming to your desired labels
    column_map_top = {
        'seen transform.': 'Seen Transformations',
        'new transform.': 'New Transformations'
    }
    column_map_bottom = {
        'seen': 'Seen Alphabets',
        'new': 'New Alphabets'
    }

    # 4. Apply the renaming to both levels of the MultiIndex
    df_pivot.columns = df_pivot.columns.set_levels([
        df_pivot.columns.levels[0].map(column_map_top), 
        df_pivot.columns.levels[1].map(column_map_bottom)
    ])

    # 5. Clean up level names and reorder columns for better readability
    df_pivot.columns.names = [None, None]
    cols_ordered = [
        ('Seen Transformations', 'Seen Alphabets'),
        ('Seen Transformations', 'New Alphabets'),
        ('New Transformations', 'Seen Alphabets'),
        ('New Transformations', 'New Alphabets')
    ]
    df_pivot = df_pivot[cols_ordered]
    df_pivot = df_pivot.reset_index(level=['filename_model', 'batching_method', 'num. seen alphabets in training'], drop=True)
    df_pivot = df_pivot.sort_values(by=('New Transformations', 'New Alphabets'))
    return df_pivot


def render_model_choice_page():
    st.header("Performance Comparison")
    df = load_performance_results()
    # Display the multi-index dataframe
    st.dataframe(df, use_container_width=True)

    few_shot_choice = st.segmented_control(
        label="Select Num. Study Examples", 
        options=list(range(1, 6)), 
        default=1
    )

    if few_shot_choice == 1:
        replication = st.number_input("Replication Run", min_value=0, max_value=4, value=0)
        model_choice = st.segmented_control(
            label="Select Model", 
            options=["Noncopy (20 alphabets)", "Copy (20 alphabets)", "Copy (200 alphabets)"], 
            default="Copy (200 alphabets)"
        )
        rep = "" if replication == 0 else f"_rep{replication}"
        if model_choice == "Noncopy (20 alphabets)":
            model_path = f"../../models/batching_experiments/MLC_batchrand_dallstudy1_nep20{rep}.pt"
        elif model_choice == "Copy (20 alphabets)": 
            model_path = f"../../models/copy_batching_experiments/MLC_batchrand_dallstudy1_copy_perm20_nep20{rep}.pt"
        elif model_choice == "Copy (200 alphabets)":
            model_path = f"../../models/num_permuted_alphabets/MLC_batchalph_dallstudy1_copy_perm200_nep20{rep}.pt" 
        else:
            raise NotImplementedError(f"{model_choice} is not a valid choice.")  
        # load constant variables across tabs
        dl = utils.get_dataloader(path="../../data/letter-string-analogies/perturbation_dataset")
        cp, model = utils.get_setup(model_path)
    
    else:
        model_path = f"../../models/study_examples/MLC_batchstudy_alph_dallstudy{few_shot_choice}_copy_perm20_nep20.pt"
        dl = utils.get_dataloader(path=f"../../data/letter-string-analogies/all_transformations_study{few_shot_choice}_copy_perm20") 
        cp, model = utils.get_setup(model_path)


    return cp, model, dl