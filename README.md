# MLC for letter-string analogies

Applying Meta-Learning for Compositionality (MLC) introduced by Lake & Baroni (2023)[^1] to letter-string analogy problems.

- The code for creating MLC models and running training is adapted from Lake & Baroni (2023)[^1] which you can find [here](https://github.com/brendenlake/MLC); the code for generating the datasets is adapted from Lewis & Mitchell (2024)[^2] which you can find [here](https://github.com/marthaflinderslewis/counterfactual_analogy/).

# Table of Contents
### [Overview](#overview-1)
- [Requirements](#requirements)
- [Repository Structure](#repository-structure)
### [Replication](#replication-1)
- [Datasets](#datasets)
- [Training](#training)
### [Experiments](#experiments-1)
- [Batching strategies](#batching-strategies)
### [References](#references-1)

## Overview 

### Requirements

- Python version 3.12
- Dependencies: 
  - `matplotlib`
  - `numpy`
  - `pandas`
  - `seaborn`
  - `torch`
  - `tqdm`
- install all packages by running the following command in your terminal:
  ```
  pip install -r requirements.txt
  ```

### Repository Structure
```
├───analyses            - analysis notebooks for trained models.
│
├───data                - datasets used for analyses
|
├───hyperparameter_search
│       config.py       - config - fixed and varied hyperparameters
│       run_hpsearch.py - run hyperparameter search
│
├───models              - Saved checkpoints that include the model parameters of trained models
|
│   checkpoint.py       - CheckPoint class - used for saving/loading models
|   datasets.py         - implements Language and Dataset classes used for dataloading
|   evaluate.py         - evaluation functions for batches given an MLC model
|   generate_data.py    - generates the dataset (problem sets of letter-string analogies)
|   model.py            - contains the MLC model class
│   requirements.txt    - required python packages
|   train.py            - training loop for the MLC model
|   timing.py           - timing utility function
```

## Replication

### Datasets
You can obtain the datasets in two ways:
1. Download them [here](link).
2. Generate datasets yourself.
    #### Code Snippets
    The snippets below generate all datasets used in our project (`data_dir` corresponds to the directory you find when downloading the data):
    ```python
    python generate_data.py --data_dir "data/all_transformations_study1"
    python generate_data.py --data_dir "data/all_transformations_study3" --study_examples 3
    python generate_data.py --data_dir "data/all_transformations_study1_incl_copy_fixed_gen" --copy

    # Dataset with new alphabets:
    python generate_data.py --data_dir "data/all_transformations_study1_new_alphabets" --n_reshuffle 5 --seed 123
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
For training to work by default, the data needs to be structured as follows:
```
├───data
│   ├───<dataset name>
│   │       train.csv
│   │       val.csv
```

For instance, to train a model for `10` epochs on the `base_problems` dataset, run:
```python
python train.py --filename_model "MLC_dbase_nep10.pt" --nepochs 10 --dir_data "data/base_problems"
```

## Experiments <a name="Experiments"></a>

### Batching strategies

Shared Arguments across experiments:
| Argument        | Value                           |
|:----------------|:--------------------------------|
| dataset         | data/all_transformations_study1 |
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

Results:
| filename_model                           | batching_method   | query_first   |   loss |   accuracy in-dist |   accuracy out-of-dist |
|:-----------------------------------------|:------------------|:--------------|-------:|-------------------:|-----------------------:|
| MLC_batchunstruct_dallstudy1_nep20.pt    | unstructured      | False         |  **1.079** |              **0.68**  |                  0.118 |
| MLC_batchunstruct_qf_dallstudy1_nep20.pt | unstructured      | True          |  1.09  |              0.623 |                  0.086 |
| MLC_batchbytrans_dallstudy1_nep20.pt     | transformation    | False         |  1.203 |              0.64  |                  **0.121** |
| MLC_batchbyalph_dallstudy1_nep20.pt      | alphabet          | False         |  1.112 |              0.618 |                  0.084 |
| MLC_batchbyboth_dallstudy1_nep20.pt      | both              | False         |  1.257 |              0.602 |                  0.098 |

Loss and Accuracy are based on the lowest/highest value achieved during training on the *validation* set. 

### Num. permuted Alphabets

- We assessed whether generalization gets better the more permuted alphabets are included in the dataset. To do this, we compared the accuracy of models trained on 4 different datasets containing 20, 40, 60, and 80 permuted alphabets.
```python
python generate_data.py --data_dir "data/all_transformations_study1_copy_perm40" --alphabets_per_perm_level 5 --copy

python generate_data.py --data_dir "data/all_transformations_study1_copy_perm40" --alphabets_per_perm_level 10 --copy
python generate_data.py --data_dir "data/all_transformations_study1_copy_perm60" --alphabets_per_perm_level 15 --copy

python generate_data.py --data_dir "data/all_transformations_study1_copy_perm80" --alphabets_per_perm_level 20 --copy
```
- To match dataset sizes we used `shrink_dataset.py` located in the `data` directory:
```python
python shrink_dataset.py --dataset "all_transformations_study1_copy_perm40" --reference_dataset "all_transformations_study1_copy_perm20"

python shrink_dataset.py --dataset "all_transformations_study1_copy_perm60" --reference_dataset "all_transformations_study1_copy_perm20"

python shrink_dataset.py --dataset "all_transformations_study1_copy_perm80" --reference_dataset "all_transformations_study1_copy_perm20"
```

## References
[^1]: Lake, B. M., & Baroni, M. (2023). Human-like systematic generalization through a meta-learning neural network. Nature, 623(7985), 115-121. https://doi.org/10.1038/s41586-023-06668-3
[^2]: Lewis, M., & Mitchell, M. (2024). Using Counterfactual Tasks to Evaluate the Generality of Analogical Reasoning in Large Language Models. Proceedings of the Annual Meeting of the Cognitive Science Society, 46. https://escholarship.org/uc/item/58d9s666
