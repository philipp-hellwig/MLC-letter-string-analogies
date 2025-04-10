import numpy as np
import pandas as pd
import string
from copy import deepcopy
import random
from itertools import chain
from tqdm import tqdm

# Generate derangement
def k_derange(k, letters):
    if k == 1:
        return None, None, letters
    
    to_shuffle = sorted(random.sample(range(len(letters)), k=k))
    shuffled = random.sample(to_shuffle, k=len(to_shuffle))

    # number of letters that have not been shuffled
    not_derangement = sum([i == shuffled[to_shuffle.index(i)] for i in to_shuffle])

    # repeat until shuffled
    while not_derangement:
        shuffled = random.sample(to_shuffle, k=len(to_shuffle))
        not_derangement = sum([i == shuffled[to_shuffle.index(i)] for i in to_shuffle])
    shuffled_alphabet = [letters[i] if i not in to_shuffle else letters[shuffled[to_shuffle.index(i)]] for i in range(len(letters))]
    return to_shuffle, [letters[i] for i in shuffled], shuffled_alphabet


# Successor transformation
def apply_succ(prob_letters, *args):
    return [prob_letters[:-1], prob_letters[:-2] + [prob_letters[-1]]]


# Predecessor transformation
def apply_pred(prob_letters, *args):
    return [prob_letters[1:], [prob_letters[0]] + prob_letters[2:]]


# Add letter to sequence
def apply_add_letter(prob_letters, *args):
    return [prob_letters[:-1], prob_letters]


# Remove redundant letter
def apply_remove_redundant(prob_letters, *args):
    redundant_loc = np.arange(len(prob_letters))
    np.random.shuffle(redundant_loc)
    redundant_loc = redundant_loc[0]
    prob_redundant = deepcopy(prob_letters)
    prob_redundant.insert(redundant_loc, prob_letters[redundant_loc])
    return [prob_redundant, prob_letters]


# Remove out-of-place character
def apply_fix_alphabet(prob_letters, letters, *args):
    remaining_letters = np.array(deepcopy(letters))
    remaining_letters = remaining_letters[np.all(np.expand_dims(np.array(remaining_letters),1) != np.expand_dims(np.array(prob_letters),0), 1)]
    np.random.shuffle(remaining_letters)
    insert_letter = remaining_letters[0]
    insert_loc = np.arange(len(prob_letters))
    np.random.shuffle(insert_loc)
    insert_loc = insert_loc[0]
    prob_letters_insert = deepcopy(prob_letters)
    prob_letters_insert[insert_loc] = insert_letter
    return [prob_letters_insert, prob_letters]


# Sort letters
def apply_sort(prob_letters, *args):
    swap_loc = np.arange(len(prob_letters))
    np.random.shuffle(swap_loc)
    i_loc = swap_loc[0]
    j_loc = swap_loc[1]
    i_letter = prob_letters[i_loc]
    j_letter = prob_letters[j_loc]
    prob_swapped = deepcopy(prob_letters)
    prob_swapped[i_loc] = j_letter
    prob_swapped[j_loc] = i_letter
    return [prob_swapped, prob_letters]


def generate_dataset(
        alphabet_permutations=[1, 2, 5, 10, 20], 
        prob_lengths=[2,3,4,5,6],
        transformations={
            'succ' : apply_succ,
            'pred' : apply_pred,
            'add_letter' : apply_add_letter,
            'remove_redundant' : apply_remove_redundant,
            'fix_alphabet' : apply_fix_alphabet,
            'sort' : apply_sort
        },
        n_examples=1,
        n_reshuffle=50
        ) -> pd.DataFrame:
    
    """Generate a dataset of letter string analogies and save them to train and val folders.
    Each text file contains a set of queries and a set of study examples and the alphabet it was generated from.

    Args:
        alphabet_permutations (list, optional): How many letters each subset . Defaults to [1, 2, 5, 10, 20].
        prob_lengths (list, optional): Which query lengths should be included. For example, query a b c -> ? has length 3. Defaults to [2,3,4,5,6].
        transformations (dict, optional): name of transformation [str]: transformation [function]. Defaults to { 'succ' : apply_succ, 'pred' : apply_pred, 'add_letter' : apply_add_letter, 'remove_redundant' : apply_remove_redundant, 'fix_alphabet' : apply_fix_alphabet, 'sort' : apply_sort }.
        n_study (int, optional): How many study examples should be included? Defaults to 3.
        n_reshuffle (int, optional): How many times to reshuffle to get new query - study example pairs.
    Returns:
        pd.DataFrame: Dataframe with columns: n_perm, alphabet, transformation, query, study.
    """
    # data set should be csv with (n_permutations, alphabet, study examples, query, problem type)
    n = 0
    # initialize data frame:
    cols = ["n_perm", "alphabet", "transformation", "query"] #+ ["study" + str(i) for i in range(1, n_study+1)]
    dataset = pd.DataFrame(columns=cols)
    for n_perm in alphabet_permutations:
        queries_by_trans = {name: [] for name in transformations.keys()}

        # permute alphabet:
        _, _, alphabet = k_derange(n_perm, list(string.ascii_lowercase))
        alph_string = "".join(alphabet)
        
        for name in transformations.keys():
            queries = []
            for length in prob_lengths:
                for i in range(len(alphabet)-length+1):
                    query = transformations[name](alphabet[i:i+length+1], alphabet)
                    query = "".join(query[0]) + "->" + "".join(query[1])
                    queries.append(query)
                    n += 1

            # generate study examples by reshuffling the queries n_reshuffle times:
            for _ in range(n_reshuffle):
                new_data = pd.DataFrame(
                    {"n_perm" : [n_perm for _ in range(len(queries))],
                    "alphabet" : [alph_string for _ in range(len(queries))],
                    "transformation" : [name for _ in range(len(queries))],
                    "query" : queries,
                    "study" : np.random.permutation(queries)
                    })
                dataset = pd.concat([dataset, new_data], ignore_index=True)
    print(f"Generated {n} unique queries.")
    # drop rows where query is the same as study example:
    dataset = dataset.loc[dataset["query"] != dataset["study"], :]
    print(f"Resulting in {dataset.shape[0]} total samples.")
    dataset.to_csv("data/all_samples.csv", index=False)
    return dataset


def dataset_to_disk(
        dataset: pd.DataFrame, 
        directory="data", 
        batch_size: int=20, 
        train_ratio: float=0.9, 
        n_support: int=3) -> None:
    
    """Write dataset to batched .csv files.

    Args:
        dataset (dict): dictionary generated from generate_dataset function. 
        directory (str): directory where the dataset should be saved. Defaults to "data".
        batch_size (int, optional): Specifies how many queries should be in each batch. Defaults to 20.
        train_ratio (float, optional): Proportion of batches allocated to train folder. Defaults to 0.9 (number should be between 0 and 1).
    """
    # shuffle dataset:
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    rows = dataset.shape[0]
    file_id = 1
    num_train_samples = round(train_ratio * rows)
    print("Writing to .csv files...")
    for i in tqdm(range(0, rows, batch_size)):
        # set subdirectory:
        if i <= num_train_samples:
            subdir= "train"
        else:
            subdir= "val"
        batch = dataset.iloc[i:i+batch_size, :]
        batch.to_csv(f"{directory}/{subdir}/{str(file_id).zfill(len(str(rows//batch_size)))}.csv", index=False)
        file_id += 1
    
    print(f"Done.")

if __name__ == "__main__":
    dataset = generate_dataset(n_reshuffle=1)
    dataset_to_disk(dataset)    
