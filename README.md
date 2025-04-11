# MLC for letter-string analogies

Applying Meta-Learning for Compositionality (MLC) introduced by Lake & Baroni (2023)[^1] to letter-string analogy problems.

- The code for creating the models and the training loop is adapted from Lake & Baroni (2023)[^1] which you can find [here](https://github.com/brendenlake/MLC); the code for generating the data set is adapted from Lewis & Mitchell (2024)[^2] which you can find [here](https://github.com/marthaflinderslewis/counterfactual_analogy/).

## Overview

### File structure

```
|   datasets.py         - implements Language and Dataset classes
|   evaluate.py         - evaluation functions for batches given an MLC model
|   generate_data.py    - generates the dataset (problem sets of letter-string analogies)
|   model.py            - contains the MLC model class
|   train.py            - training loop for the MLC model
|   train_lib.py
```
## Replication

### Dataset
To obtain the same dataset, either run
```python
python generate_data.py
```
or download the dataset from [here](link).

### Training
For training to work by default, the data needs to be structured as follows:
```
+---data
|       train.csv
|       val.csv
```

To train the model, run
```python
python train.py --fn_out_model "MLC-model.pt" --nepochs 10
```

[^1]: Lake, B. M., & Baroni, M. (2023). Human-like systematic generalization through a meta-learning neural network. Nature, 623(7985), 115-121. https://doi.org/10.1038/s41586-023-06668-3
[^2]: Lewis, M., & Mitchell, M. (2024). Using Counterfactual Tasks to Evaluate the Generality of Analogical Reasoning in Large Language Models. Proceedings of the Annual Meeting of the Cognitive Science Society, 46. https://escholarship.org/uc/item/58d9s666
