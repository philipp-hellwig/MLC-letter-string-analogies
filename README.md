# MLC for letter-string analogies

Applying Meta-Learning for Compositionality (MLC) introduced by Lake & Baroni (2023)[^1] to letter-string analogy problems.

- The code for creating the models and the training loop is adapted from Lake & Baroni (2023)[^1] which you can find [here](https://github.com/brendenlake/MLC); the code for generating the data set is adapted from Lewis & Mitchell (2024)[^2] which you can find [here](https://github.com/marthaflinderslewis/counterfactual_analogy/).

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

### Repository Descripiton

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
1. to get all datasets used in the paper, download them [here](link).
2. Generate datasets yourself.
    - For example, to obtain a dataset run:
    ```sh
    python generate_data.py --transformations "base" --n_reshuffle 1 --alphabets_per_permutation 1 --data_dir "data/base_tiny"
    ```
    #### Arguments
    ```sh
    --transformations TRANSFORMATIONS
                        Comma-separated list of integers of transformations to include (e.g., 1,2 will include extend sequence and successor). Defaults to all.
    
    --data_dir DATA_DIR   The directory in which the data set will be saved
    
    --n_reshuffle N_RESHUFFLE
                            How many times to reshuffle the data to get new problem-study example pairs. Defaults to 50.
    
    --alphabets_per_permutation ALPHABETS_PER_PERMUTATION
                            How many unique alphabets to generate per permutation level. Defaults to 1.
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
```sh
python train.py --filename_model "MLC_dbase_nep10.pt" --nepochs 10 --dir_data "data/base_problems"
```
### References
[^1]: Lake, B. M., & Baroni, M. (2023). Human-like systematic generalization through a meta-learning neural network. Nature, 623(7985), 115-121. https://doi.org/10.1038/s41586-023-06668-3
[^2]: Lewis, M., & Mitchell, M. (2024). Using Counterfactual Tasks to Evaluate the Generality of Analogical Reasoning in Large Language Models. Proceedings of the Annual Meeting of the Cognitive Science Society, 46. https://escholarship.org/uc/item/58d9s666
