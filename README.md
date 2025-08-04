# MLC for letter-string analogies

Applying Meta-Learning for Compositionality (MLC) introduced by Lake & Baroni (2023)[^1] to letter-string analogy problems.

- The code for creating MLC models and running training is adapted from Lake & Baroni (2023)[^1] which you can find [here](https://github.com/brendenlake/MLC); the code for generating the datasets is adapted from Lewis & Mitchell (2024)[^2] which you can find [here](https://github.com/marthaflinderslewis/counterfactual_analogy/).


## Repository Overview 

```
├───analyses
│   │
│   ├───alternate_rule_errors.ipynb - Error analysis for noncopy model 
│   │                               - Table 3
│   │ 
│   ├───analysis_utils.py           - utilities for plotting
│   │ 
│   ├───study_examples.ipynb        - num. study examples experiments:
│   │                               - Supplementary Material Figure 1
│   │ 
│   ├───model_comparisons.ipynb     - Main results experiments (Table 1)
│   │ 
│   ├───num_perm_alphabets.ipynb    - Comparisons for varying num. seen alphabets during training
│   │                               - Figure 4, Table 2 
│   │ 
│   ├───encoder_attn_and_repr.ipynb - Interpretability Analyses for the different encoders
│   │                               - Figure 1, 5, 6
│
├───data                      
|
├───models                    - Saved checkpoints
|
│   checkpoint.py             - CheckPoint class - used for saving/loading models
|   datasets.py               - implements Language, Dataset and Dataloader classes
|   evaluate.py               - evaluation functions for batches given an MLC model
|   generate_data.py          - generates the dataset (problem sets of letter-string analogies)
|   model.py                  - contains the MLC model class
|   train.py                  - training loop for the MLC model
|   timing.py                 - timing utility function
```


## Replication

### Requirements

- Python `3.12`
- Dependencies: 
  - `matplotlib`
  - `numpy`
  - `pandas`
  - `seaborn`
  - `torch`
  - `tqdm`
- Install all packages by running the following command in your terminal:
  ```
  pip install -r requirements.txt
  ```

### Datasets

The snippets below generate all datasets used in the main experiments of the project where `data_dir` corresponds to the directory when downloading them:
```python
# Base datasets:
python generate_data.py --data_dir "data/all_transformations_study1_perm20"
python generate_data.py --data_dir "data/all_transformations_study1_copy_perm20" --copy

# Datasets varying the number of study examples:
python generate_data.py --data_dir "data/all_transformations_study2_copy_perm20" --study_examples 2 --copy
# --study_examples 3, 4, 5 for the rest of the datasets

# Datasets varying the number of seen alphabets:
python generate_data.py --data_dir "data/all_transformations_study1_copy_perm40" --alphabets_per_perm_level 10 --copy
# --alphabets_per_perm_level 50 for 200 seen alphabets

# Dataset with new alphabets:
python generate_data.py --data_dir "data/all_transformations_study1_new_alphabets" --n_reshuffle 5 --seed 123
```

- To keep dataset sizes constant we used `shrink_dataset.py` located in the `data` directory for datasets that contained more than 20 permuted alphabets.
For example:
```python
python shrink_dataset.py --dataset "all_transformations_study1_copy_perm40" --reference_dataset "all_transformations_study1_copy_perm20"
```

#### Arguments
```    
--data_dir 
      The directory in which the data set will be saved. Default: "data/debug".

--transformations 
      "base", "all" or Comma-separated list of integers of transformations to include 
      (e.g., 1,2 will include extend sequence and successor). Default: "all".

--n_reshuffle 
      How many times to reshuffle the data to get new problem-study example pairs. Default: 10.

--alphabets_per_permutation 
      How many unique alphabets to generate per permutation level. Default: 5.

--study_examples
      How many study examples to show per problem. Default: 1.
```


### Training

Experiments can be run using `train.py` (GPU access required for reasonable computation times).
For instance, to run a model for 20 epochs with batches of size 32 grouped by alphabet on a letter-string dataset run:
```python
python train.py --filename_model "MLC_batchbyalph_dallstud1_nep20.pt" --nepochs 20 --batch_size 32 --dir_data "data/all_transformations_study1_perm20" --batching_method "alphabet"
```

All experiments presented in the paper shared the following parameters:

| Argument        | Value                           |
|:----------------|:--------------------------------|
| batch_size      | 32                              |
| nepochs         | 20                              |
| lr              | 0.001                           |
| lr_end_factor   | 0.05                            |
| lr_warmup       | True                            |
| nlayers_encoder | 3                               |
| nlayers_decoder | 3                               |
| nheads          | 8                               |
| emb_size        | 128                             |
| ff_mult         | 4                               |
| dropout         | 0.1                             |
| act             | gelu                            |

To reproduce the results of the experiments in the paper, you can run the snippets shown in each section below.
#### Batching Methods
```python
# Batching method random:
python train.py --filename_model "MLC_batchrand_dallstud1_nep20.pt" --nepochs 20 --batch_size 32 --dir_data "data/all_transformations_study1_perm20" --batching_method "random"

# Batching method alphabet:
python train.py --filename_model "MLC_batchbyalph_dallstudy1_nep20.pt" --nepochs 20 --batch_size 32 --dir_data "data/all_transformations_study1_perm20" --batching_method "alphabet"

# Batching method transformation:
python train.py --filename_model "MLC_batchbytrans_dallstud1_nep20.pt" --nepochs 20 --batch_size 32 --dir_data "data/all_transformations_study1_perm20" --batching_method "transformation"

# Batching method transformation and alphabet:
python train.py --filename_model "MLC_batchbyboth_dallstud1_nep20.pt" --nepochs 20 --batch_size 32 --dir_data "data/all_transformations_study1_perm20" --batching_method "transformation_alphabet"
```

#### Copy Tasks
```python
# Batching method random:
python train.py --filename_model "MLC_batchrand_dallstudy1_copy_perm20_nep20.pt" --nepochs 20 --batch_size 32 --dir_data "data/all_transformations_study1_copy_perm20" --batching_method "random"

# Batching method alphabet:
python train.py --filename_model "MLC_batchalph_dallstudy1_copy_perm20_nep20.pt" --nepochs 20 --batch_size 32 --dir_data "data/all_transformations_study1_copy_perm20" --batching_method "alphabet"

# Batching method transformation:
python train.py --filename_model "MLC_batchbyboth_dallstud1_nep20.pt" --nepochs 20 --batch_size 32 --dir_data "data/all_transformations_study1_copy_perm20" --batching_method "transformation"

# Batching method transformation and alphabet:
python train.py --filename_model "MLC_batchtrans_dallstud1_nep20.pt" --nepochs 20 --batch_size 32 --dir_data "data/all_transformations_study1_copy_perm20" --batching_method "transformation_alphabet"
```

#### Number of Training Alphabets
```python
# Batching method random for 40 permuted alphabets in training:
python train.py --filename_model "MLC_batchrand_dallstudy1_copy_perm40_nep20.pt" --nepochs 20 --batch_size 32 --dir_data "data/all_transformations_study1_copy_perm40" --batching_method "random"

# Batching method alphabet for 40 permuted alphabets in training:
python train.py --filename_model "MLC_batchalph_dallstudy1_copy_perm40_nep20.pt" --nepochs 20 --batch_size 32 --dir_data "data/all_transformations_study1_copy_perm40" --batching_method "alphabet"

# for more training alphabets, change 40 to the desired number (granted the corresponding datasets exists)
```


## References
[^1]: Lake, B. M., & Baroni, M. (2023). Human-like systematic generalization through a meta-learning neural network. Nature, 623(7985), 115-121. https://doi.org/10.1038/s41586-023-06668-3
[^2]: Lewis, M., & Mitchell, M. (2024). Using Counterfactual Tasks to Evaluate the Generality of Analogical Reasoning in Large Language Models. Proceedings of the Annual Meeting of the Cognitive Science Society, 46. https://escholarship.org/uc/item/58d9s666
