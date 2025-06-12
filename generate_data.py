import argparse
import ast
from copy import deepcopy
import os
import random
import string

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------------------------
#                                  Overview of Transformations
# ---------------------------------------------------------------------------------------------
# Each problem function returns a list of length 2 [source sequence, transformation]
#
#  Transformations present in both data splits
# 1.	Extend Sequence (a b c d -> a b c d e)
# 2.	Successor (a b c d -> a b c e)
# 3.	Predecessor (b c d e -> a c d e)
# 4.	Remove Redundant letter (a b b c d -> a b c d)
# 5.	Fix alphabetic sequence (a b c w e -> a b c d e)
# 6.	Sort (a d c b e -> a b c d e)
# 7.	Sort + Group (aa dd cc bb ee -> aa bb cc dd ee)
# 8.	Remove redundant + Interleave (a x b x b x c x d -> a x b x c x d)
# 9.	Remove Redundant + successor (a b b c d -> a b c e)
# 10.	Fix alphabetic sequence + extend sequence (a b c w -> a b c d e)
#
# Transformations only present in validation/ test splits:
# 11.   Remove Redundant + Sort (a d d c b e -> a b c d e)
# 12.   Extend Sequence + Predecessor (b c d e -> a c d e f)
# 13.   Fix alphabetic sequence + Interleave (a f b f c f w f e -> a f b f c f d f e)
# 14.   Extend sequence + Group (aa bb cc dd -> aa bb cc dd ee)
# 15.   Extend Sequence + Extend Sequence + Successor (a b c d -> a b c d e g)
# 16.   Fix alphabetic Sequence + Predecessor + Successor (a b c w e -> a a c d f)
# 17.   Reverse (a b c d -> d c b a)
# 18.   Shift (a b c d -> e f g h)
# 19.   Replicate (a b c d -> a b c d a b c d)
# ---------------------------------------------------------------------------------------------


# 1. Append letter to sequence
def extend_sequence(prob_letters, alphabet, *args):
    idx_last = alphabet.index(prob_letters[-1])
    if idx_last == (len(alphabet)-1):
        raise IndexError("Letter can't be added to a sequence ends with the last letter of the alphabet!")
    return [prob_letters, prob_letters + [alphabet[idx_last+1]]]


# 2 Successor transformation
def succ(prob_letters, alphabet, *args):
    # find index in alphabet of last problem letter
    idx_last = alphabet.index(prob_letters[-1])
    if idx_last == (len(alphabet)-1):
        raise IndexError("Successor transformation cannot be applied to a problem that ends with the last letter of the alphabet!")
    return [prob_letters, prob_letters[:-1] + [alphabet[idx_last+1]]]


# 3 Predecessor transformation
def pred(prob_letters, alphabet, *args):
    # find index in alphabet of first problem letter
    idx_first = alphabet.index(prob_letters[0])
    if idx_first == 0:
        raise IndexError("Predecessor transformation cannot be applied to a problem that starts with the first letter of the alphabet!")
    return [prob_letters, [alphabet[idx_first-1]] + prob_letters[1:]]


# 4 Remove redundant letter
def remove_redundant(prob_letters, *args):
    redundant_loc = np.arange(len(prob_letters))
    np.random.shuffle(redundant_loc)
    redundant_loc = redundant_loc[0]
    prob_redundant = deepcopy(prob_letters)
    prob_redundant.insert(redundant_loc, prob_letters[redundant_loc])
    return [prob_redundant, prob_letters]


# 5 Remove out-of-place character
def fix_alphabetic_seq(prob_letters, alphabet, *args):
    remaining_letters = np.array(deepcopy(alphabet))
    remaining_letters = remaining_letters[np.all(np.expand_dims(np.array(remaining_letters),1) != np.expand_dims(np.array(prob_letters),0), 1)]
    np.random.shuffle(remaining_letters)
    insert_letter = remaining_letters[0]
    insert_loc = np.arange(len(prob_letters))
    np.random.shuffle(insert_loc)
    insert_loc = insert_loc[0]
    prob_letters_insert = deepcopy(prob_letters)
    prob_letters_insert[insert_loc] = insert_letter
    return [prob_letters_insert, prob_letters]


# 6 Sort letters
def sort(prob_letters, *args):
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


# group a problem and its solution by duplicating each letter k times
def group(problem: list, k: int=None, **kwargs):
    """Transform problem by repeating each letter k times.

    Args:
        problem (list): [[source],[transformation]]
        k (int): how many times to repeat each letter
    """
    p = deepcopy(problem)
    if k is None:
        k = np.random.randint(2,4)
    source = list(np.repeat(p[0], k))
    trans = list(np.repeat(p[1], k))
    return [source, trans]


# interleave a problem and its solution
def interleave(problem: list, alphabet: list, interleaver: str=None, **kwargs):
    p = deepcopy(problem)
    if interleaver is None:
        interleaver = np.random.choice(alphabet)
    source = []
    for i, letter in enumerate(p[0]):
        source.append(letter)
        if i < (len(p[0])-1):
            source.append(interleaver)
    trans = []
    for i, letter in enumerate(p[1]):
        trans.append(letter)
        if i < (len(p[1])-1):
            trans.append(interleaver)
    return [source, trans]


# 9. Remove Redundant + successor (a b b c d -> a b c e)
def rr_succ(prob_letters, *args):
    redundant_prob = remove_redundant(prob_letters)
    succ_prob = succ(redundant_prob[1], *args)
    return [redundant_prob[0], succ_prob[1]]


# 10. Fix alphabetic sequence + extend sequence (a b c w -> a b c d -> a b c d e)
def fix_extend(prob_letters, *args):
    fix_seq = fix_alphabetic_seq(prob_letters, *args)
    extend_seq = extend_sequence(fix_seq[1], *args)
    return [fix_seq[0], extend_seq[1]]


# 11. Remove Redundant + Sort (a d d c b e -> a b c d e)
def rr_sort(prob_letters, *args):
    redundant_prob = remove_redundant(prob_letters)
    sort_prob = sort(redundant_prob[0], *args)
    return [sort_prob[0], redundant_prob[1]]


# 12. Extend Sequence + Predecessor (b c d e -> a c d e f)
def extend_pred(prob_letters, *args):
    return [prob_letters, pred(extend_sequence(prob_letters, *args)[1], *args)[1]]


# 14. Extend sequence + Group (aa bb cc dd -> aa bb cc dd ee)
def extend_group(prob_letters, *args):
    return group(extend_sequence(prob_letters, *args))


# 15. Extend Sequence + Extend Sequence + Successor (a b c d -> a b c d e g)
def extend_extend_succ(prob_letters, *args):
    trans = succ(extend_sequence(extend_sequence(prob_letters, *args)[1], *args)[1], *args)[1]
    return [prob_letters, trans]


# 16. Fix alphabetic Sequence + Predecessor + Successor (a b c w e -> a a c d f)
def fix_pred_succ(prob_letters, *args):
    source = fix_alphabetic_seq(prob_letters, *args)[0]
    trans = succ(pred(source, *args)[1], *args)[1]
    return [source, trans]


# 17. Reverse (a b c d -> d c b a)
def reverse(prob_letters, *args):
    return [prob_letters, [letter for letter in reversed(prob_letters)]]


# 18. Shift (a b c d -> e f g h)
def shift(prob_letters, alphabet, *args):
    idx_last = alphabet.index(prob_letters[-1])
    if idx_last > (len(alphabet)-1-len(prob_letters)):
        raise IndexError("Sequence is too close to the end of the alphabet to be shifted")
    return [prob_letters, alphabet[idx_last+1:idx_last+1+len(prob_letters)]]


# 19. Replicate (a b c d -> a b c d a b c d)
def replicate(prob_letters, *args):
    return [prob_letters, list(np.tile(prob_letters, 2))]


# permutes alphabet
def k_derange(k, alphabet):
    if k in [0,1]:
        return None, None, alphabet
    
    to_shuffle = sorted(random.sample(range(len(alphabet)), k=k))
    shuffled = random.sample(to_shuffle, k=len(to_shuffle))

    # number of letters that have not been shuffled
    not_derangement = sum([i == shuffled[to_shuffle.index(i)] for i in to_shuffle])

    # repeat until shuffled
    while not_derangement:
        shuffled = random.sample(to_shuffle, k=len(to_shuffle))
        not_derangement = sum([i == shuffled[to_shuffle.index(i)] for i in to_shuffle])
    shuffled_alphabet = [alphabet[i] if i not in to_shuffle else alphabet[shuffled[to_shuffle.index(i)]] for i in range(len(alphabet))]
    return to_shuffle, [alphabet[i] for i in shuffled], shuffled_alphabet


def get_unique_problems_counterfactual_analogy():
    """Returns unique problems used in Lewis, M., & Mitchell, M. (2024).
    doi: https://doi.org/10.48550/arXiv.2402.08955
    """
    lewis_mitchell = pd.read_csv("https://raw.githubusercontent.com/marthaflinderslewis/counterfactual_analogy/refs/heads/main/data/data.csv")
    lewis_mitchell["target_1"] = lewis_mitchell["target_1"].apply(ast.literal_eval)
    lewis_mitchell["correct_answer"] = lewis_mitchell["correct_answer"].apply(ast.literal_eval)
    lewis_mitchell["sep"] = [[">"] for _ in range(lewis_mitchell.shape[0])]
    lewis_mitchell["problem"] = lewis_mitchell['target_1'] + lewis_mitchell["correct_answer"]
    lewis_mitchell["problem"] = lewis_mitchell['target_1'] + lewis_mitchell["sep"] + lewis_mitchell["correct_answer"]
    lewis_mitchell["problem"] = lewis_mitchell["problem"].apply(lambda x: " ".join(x))
    return lewis_mitchell["problem"].unique()


def get_copy_study_examples(row, study_by_alph_trans):
    dim_study_examples = study_by_alph_trans[row["alphabet"]][row["transformation"]].shape
    study_idx = np.random.randint(dim_study_examples[1])
    # study example to replace with the actual example:

    study_examples = study_by_alph_trans[row["alphabet"]][row["transformation"]][:,study_idx]
    replace_idx = np.random.randint(dim_study_examples[0])
    study_examples[replace_idx] = row["problem"]
    study_examples = " | ".join(study_examples)
    row["study"] = study_examples
    return row


def generate_dataset(
        transformations: dict,
        permutation_levels: list=[0, 2, 5, 10, 20], 
        prob_lengths: list=[2,3,4,5,6],
        n_reshuffle: int=50,
        alphabets_per_perm_level: int=1,
        n_study: int=3,
        prop_study: float=1.0,
        copy=False
        ) -> pd.DataFrame:
    
    """Generate a dataset of letter string analogies and save them to train and val folders.
    Each text file contains a set of queries and a set of study examples and the alphabet it was generated from.

    Args:
        transformations dict: integer of transformation [int]: transformation [function].
        permutation_levels (list, optional): How many letters each subset . Default: [0, 2, 5, 10, 20].
        prob_lengths (list, optional): Which query lengths should be included. For example, query a b c -> ? has length 3. Default: [2,3,4,5,6].
        n_study (int, optional): How many study examples should be included? Default: 1.
        n_reshuffle (int, optional): How many times to reshuffle to get new query-study example pairs. Default: 50.
        copy (bool, optional): Whether to include an example for each query where the study example (query + solution) is the same as the query. Default: False.
    Returns:
        pd.DataFrame: Dataframe with columns: n_perm, alphabet, transformation, query, study.
    """
    assert(prop_study > 0 and prop_study <=1)
    # data set should be csv with (n_permutations, alphabet, study examples, query, problem type)
    n = 0
    # initialize data frame:
    cols = ["n_perm", "alphabet", "transformation", "problem", "query_length"] #+ ["study" + str(i) for i in range(1, n_study+1)]
    dataset = pd.DataFrame(columns=cols)
    study_by_alph_trans = {}
    for n_perm in tqdm(permutation_levels, desc="Generating Problems"):
        alphabets_n_perm = []
        for _ in range(alphabets_per_perm_level):
            while True:
                # permute alphabet:
                _, _, alphabet = k_derange(n_perm, list(string.ascii_lowercase))
                alph_string = " ".join(alphabet)
                if alph_string not in alphabets_n_perm:
                    alphabets_n_perm.append(alph_string)
                    break 
            study_by_alph_trans[alph_string] = {}

            for idx in transformations.keys():
                problems, query_lengths = [], []
                for length in prob_lengths:
                    for i in range(len(alphabet)-length+1):
                        try:
                            query, solution = transformations[idx]["function"](alphabet[i:i+length+1], alphabet)
                            problem = " ".join(query) + " > " + " ".join(solution)
                            problems.append(problem)
                            query_lengths.append(len(query))
                            n += 1
                        except IndexError:
                            pass
                # generate study examples by reshuffling the problems n_reshuffle times:
                reshuffled_datasets = []
                study_tasks = deepcopy(problems)
                if prop_study < 1:
                    study_tasks = np.random.choice(study_tasks, size= int(prop_study * len(problems)))
                    study_tasks = np.random.choice(study_tasks, size=len(problems), replace=True)
                study_examples = np.array([np.random.permutation(study_tasks) for _ in range(n_study)])
                study_examples_joined = np.array([" | ".join(study_examples[:, i]) for i in range(study_examples.shape[1])])
                # apply generalization (group, interleave) to problem if applicable:
                if transformations[idx]["generalization_function"] is not None:
                    generalized_problems = []
                    for problem in problems:
                        query, solution = problem.split(" > ")
                        query = query.split()
                        solution = solution.split()
                        query, solution = transformations[idx]["generalization_function"]([query, solution], alphabet=alphabet)
                        problem = " ".join(query) + " > " + " ".join(solution)
                        generalized_problems.append(problem)
                    problems = generalized_problems
                
                study_by_alph_trans[alph_string][transformations[idx]["transformation"]] = study_examples
                for _ in range(n_reshuffle):
                    reshuffled_data = pd.DataFrame({
                            "n_perm" : [n_perm for _ in range(len(problems))],
                            "alphabet" : [alph_string for _ in range(len(problems))],
                            "transformation" : [transformations[idx]["transformation"] for _ in range(len(problems))],
                            "study" : np.random.permutation(study_examples_joined),
                            "problem" : problems,
                            "query_length" : query_lengths,
                            "generalization_type": [transformations[idx]["generalization_type"] for _ in range(len(problems))]
                        })
                    reshuffled_datasets.append(reshuffled_data)
                dataset = pd.concat([dataset] + reshuffled_datasets, ignore_index=True)
            if n_perm in [0,1]:
                break
    # drop rows where problem is one of the study examples:
    dataset = dataset[~dataset.apply(lambda row: row["problem"] in row["study"], axis=1)]
    dataset["copy"] = False
    if copy:
        copy_ds = dataset.copy(deep=True)
        copy_ds["study"] = copy_ds.apply(lambda row: get_copy_study_examples(row, study_by_alph_trans)["study"], axis=1)
        copy_ds["copy"] = True
        dataset = pd.concat([dataset, copy_ds])
    return dataset


def dataset_to_disk(
        dataset: pd.DataFrame, 
        train_on: list,
        query_overlap=False,
        directory="data", 
        train_prop: float=0.8,
        seed: int=42
        ) -> None:
    """Splits a dataset generated with `generate_dataset` into train, val and test set and writes them to .csv files

    Args:
        dataset (pd.DataFrame): dataframe generated with `generate_dataset`.
        train_on (list[str]): Which transformations should be included in the training set? list of strings where strings are function names.
        directory (str, optional): Folder in which to store the data. Defaults to "data".
        train_prop (float, optional): Proportion of data allocated to the training set. Defaults to 0.8.
    """
    # get lewis mitchell problems and remove them from dataset:
    lewis_mitchell_problem_set = get_unique_problems_counterfactual_analogy()
    dataset[~dataset["problem"].isin(lewis_mitchell_problem_set)]
    dataset["distribution"] = dataset["transformation"].apply(lambda x: "in" if x in train_on else "out-of")
    # shuffle dataset:
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)

    if query_overlap:
        total_n = dataset.shape[0]
        train_max = int(total_n*train_prop)
        train = dataset.loc[:train_max,:]
        # exclude transformation types that are not in train_on:
        train = train[train.transformation.isin(train_on)]
        val_max = int(train_max+(1-train_prop)*total_n/2)
        val = dataset.loc[train_max:val_max,:]
        test = dataset.loc[val_max:,:]
        train_problem_study = (train["problem"] + train["study"]).unique()
        val = val[~(val.problem + val.study).isin(train_problem_study)]
        test = test[~(test.problem + test.study).isin(train_problem_study)]
    
    else:
        unique_problems = dataset.problem.unique()
        np.random.shuffle(unique_problems)
        train_max = int(len(unique_problems) * train_prop)
        train_problems = unique_problems[:train_max]

        train = dataset[dataset.problem.isin(train_problems)]
        # exclude transformation types that are not in train_on:
        train = train[train.transformation.isin(train_on)]

        val_max = train_max + (len(unique_problems)-train_max) // 2
        val_problems = unique_problems[train_max:val_max]
        val = dataset[dataset.problem.isin(val_problems)]
        test_problems = unique_problems[val_max:]
        test = dataset[dataset.problem.isin(test_problems)]

        # resize val and test if train set got smaller due to filtering:
        n_nontest = train.shape[0] / train_prop * (1-train_prop)/2
        if n_nontest < val.shape[0]:
            val = val.sample(frac=n_nontest/val.shape[0], random_state=seed).reset_index(drop=True)
        if n_nontest < test.shape[0]:
            test = test.sample(frac=n_nontest/test.shape[0], random_state=seed).reset_index(drop=True)
    
    # save datasets as csv:
    train.to_csv(f"{directory}/train.csv", index=False)
    val.to_csv(f"{directory}/val.csv", index=False)
    test.to_csv(f"{directory}/test.csv", index=False)
    n = train.shape[0] + val.shape[0] + test.shape[0]
    print(f"{train.shape[0]:,} ({train.shape[0]/n*100:.1f}%) training-, {val.shape[0]:,} ({val.shape[0]/n*100:.1f}%) validation-, and {test.shape[0]:,} ({test.shape[0]/n*100:.1f}%) test problems written to disk.\nDone.")


ALL_TRANSFORMATIONS = {
    1: {
        "transformation": "extend_sequence",
        "function": extend_sequence,
        "generalization_function": None,
        "generalization_type": 0
    },
    2: {
        "transformation": "succ",
        "function": succ,
        "generalization_function": None,
        "generalization_type": 0
    },
    3: {
        "transformation": "pred",
        "function": pred,
        "generalization_function": None,
        "generalization_type": 0
    },
    4: {
        "transformation": "remove_redundant",
        "function": remove_redundant,
        "generalization_function": None,
        "generalization_type": 0
    },
    5: {
        "transformation": "fix_alphabetic_seq",
        "function": fix_alphabetic_seq,
        "generalization_function": None,
        "generalization_type": 0
    },
    6: {
        "transformation": "sort",
        "function": sort,
        "generalization_function": None,
        "generalization_type": 0
    },
    7: {
        "transformation": "sort_group",
        "function": sort,
        "generalization_function": group,
        "generalization_type": 0
    },
    8: {
        "transformation": "rr_interleave",
        "function": remove_redundant,
        "generalization_function": interleave,
        "generalization_type": 0
    },
    9: {
        "transformation": "rr_succ",
        "function": rr_succ,
        "generalization_function": None,
        "generalization_type": 0
    },
    10: {
        "transformation": "fix_extend",
        "function": fix_extend,
        "generalization_function": None,
        "generalization_type": 0
    },
    # val/test only transformations
    11: {
        "transformation": "rr_sort",
        "function": rr_sort,
        "generalization_function": None,
        "generalization_type": 2
    },
    12: {
        "transformation": "extend_pred",
        "function": extend_pred,
        "generalization_function": None,
        "generalization_type": 2
    },
    13: {
        "transformation": "fix_interleave",
        "function": fix_alphabetic_seq,
        "generalization_function": interleave,
        "generalization_type": 2
    },
    14: {
        "transformation": "extend_group",
        "function": extend_sequence,
        "generalization_function": group,
        "generalization_type": 2
    },
    15: {
        "transformation": "extend_extend_succ",
        "function": extend_extend_succ,
        "generalization_function": None,
        "generalization_type": 2
    },
    16: {
        "transformation": "fix_pred_succ",
        "function": fix_pred_succ,
        "generalization_function": None,
        "generalization_type": 2
    },
    17: {
        "transformation": "reverse",
        "function": reverse,
        "generalization_function": None,
        "generalization_type": 3
    },
    18: {
        "transformation": "shift",
        "function": shift,
        "generalization_function": None,
        "generalization_type": 3
    },
    19: {
        "transformation": "replicate",
        "function": replicate,
        "generalization_function": None,
        "generalization_type": 3
    }
}


def demo(sequence: list, alphabet: list=list(string.ascii_lowercase)):
    print(f"Examples with sequence {' '.join(sequence)}")
    print(f"Using alphabet: {' '.join(alphabet)}\n")
    for key, trans in ALL_TRANSFORMATIONS.items():
        print(f"{key}. {trans.__name__}:")
        query, target = trans(sequence, alphabet)
        print(f"{' '.join(query)} -> {' '.join(target)}\n")


def get_transformations(trans: str) -> dict:
    match trans: 
        case "all":
            transformations = ALL_TRANSFORMATIONS
        case "base":
            transformations = {idx: ALL_TRANSFORMATIONS[idx] for idx in range(1, 7)}
        case "train_default":
            transformations = {idx: ALL_TRANSFORMATIONS[idx] for idx in range(1, 11)}
        case _:
            func_ids = [int(idx) for idx in trans.split(',')]
            transformations = {idx: ALL_TRANSFORMATIONS[idx] for idx in func_ids}
    return transformations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="data/debug", help='The directory in which the data set will be saved')
    parser.add_argument('--transformations', default="all", help='Comma-separated list of integers of transformations to include when generating data (e.g., 1,2 will include extend sequence and successor). Defaults to all.')
    parser.add_argument('--training_transformations', default="train_default", help='Which transformations to include in training set.')
    parser.add_argument('--permutation_levels', default="1,2,5,10,20", help="Comma-seperated list of integers of permutation levels")
    parser.add_argument('--n_reshuffle', default=10, type=int, help='How many times to reshuffle the data to get new problem-study example pairs. Defaults to 10.')
    parser.add_argument('--alphabets_per_perm_level', default=5, type=int, help='How many unique alphabets to generate per permutation level. Defaults to 5.')
    parser.add_argument('--n_study', default=1, type=int, help='How many study examples to show per problem. Default is 1.')
    parser.add_argument('--prop_study', default=1.0, type=float, help='Proportion of queries to use as study examples. Default is 1.')
    parser.add_argument('--query_overlap', default=False, action='store_true', help="Whether or not to allow the same queries (with different study examples) to appear in all training splits.")
    parser.add_argument('--copy', default=False, action='store_true', help="Whether or not to include copy only tasks (examples where query is included in the study examples).")
    parser.add_argument('--seed', default=42, type=int, help="random seed for data generation. Default: 42")
    args = parser.parse_args()
    
    # set seeds for reproducibility:
    np.random.seed(args.seed)
    random.seed(args.seed)
    transformations = get_transformations(args.transformations)
    perm_levels = [int(lvl) for lvl in args.permutation_levels.split(",")]
    dataset = generate_dataset(
        transformations=transformations,
        permutation_levels=perm_levels,
        n_reshuffle=args.n_reshuffle,
        alphabets_per_perm_level=args.alphabets_per_perm_level,
        n_study=args.n_study,
        prop_study=float(args.prop_study),
        copy=args.copy
    )
    # create directory if it doesnt exist yet
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
        print(f"Created '{args.data_dir}' directory.")

    # get function names that are allowed in training set:
    train_transformations = get_transformations(args.training_transformations)
    train_transformation_names = [train_transformations[key]["transformation"] for key in train_transformations]
    # write dataset splits
    dataset_to_disk(
        dataset, 
        train_on=train_transformation_names, 
        query_overlap=args.query_overlap, 
        directory=args.data_dir, 
        seed=args.seed
    )


if __name__ == "__main__":
    main()
