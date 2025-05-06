import argparse
import ast
from copy import deepcopy
import os
import random
import string

import numpy as np
import pandas as pd
from tqdm import tqdm


# TODO: Should we let predecessor and successor problems spill over? e.g., a b c d -> z b c d or x y z -> x y a
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
# 11. Remove Redundant + Sort (a d d c b e -> a b c d e)
# 12. Extend Sequence + Predecessor (b c d e -> a c d e f)
# 13. Fix alphabetic sequence + Interleave (a f b f c f w f e -> a f b f c f d f e)
# 14. Extend sequence + Group (aa bb cc dd -> aa bb cc dd ee)
# 15. Extend Sequence + Extend Sequence + Successor (a b c d -> a b c d e g)
# 16. Fix alphabetic Sequence + Predecessor + Successor (a b c w e -> a a c d f)
# 17. Reverse (a b c d -> d c b a)
# 18. Shift (a b c d -> e f g h)
# 19. Replicate (a b c d -> a b c d a b c d)
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
def group_problem(problem: list, k: int=None, *args):
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
def interleave_problem(problem: list, alphabet: list, interleaver: str=None):
    if interleaver is None:
        interleaver = np.random.choice(alphabet)
    source = []
    for i, letter in enumerate(problem[0]):
        source.append(letter)
        if i < (len(problem[0])-1):
            source.append(interleaver)
    trans = []
    for i, letter in enumerate(problem[1]):
        trans.append(letter)
        if i < (len(problem[1])-1):
            trans.append(interleaver)
    return [source, trans]


# 7. sort and group
def sort_group(prob_letters, *args):
    return group_problem(sort(prob_letters))


# 8. remove redundant, interleave
def rr_interleave(prob_letters, *args):
    return interleave_problem(remove_redundant(prob_letters), *args)


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


# 13. Fix alphabetic sequence + Interleave (a f b f c f w f e -> a f b f c f d f e)
def fix_interleave(prob_letters, *args):
    return interleave_problem(fix_alphabetic_seq(prob_letters, *args), *args)


# 14. Extend sequence + Group (aa bb cc dd -> aa bb cc dd ee)
def extend_group(prob_letters, *args):
    return group_problem(extend_sequence(prob_letters, *args))


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

# used to permute alphabet
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


def generate_dataset(
        transformations: dict,
        alphabet_permutations: list=[1, 2, 5, 10, 20], 
        prob_lengths: list=[2,3,4,5,6],
        n_reshuffle: int=50,
        alphabets_per_permutation: int=1,
        n_study: int=3
        ) -> pd.DataFrame:
    
    """Generate a dataset of letter string analogies and save them to train and val folders.
    Each text file contains a set of queries and a set of study examples and the alphabet it was generated from.

    Args:
        transformations dict: integer of transformation [int]: transformation [function].
        alphabet_permutations (list, optional): How many letters each subset . Defaults to [1, 2, 5, 10, 20].
        prob_lengths (list, optional): Which query lengths should be included. For example, query a b c -> ? has length 3. Defaults to [2,3,4,5,6].
        n_study (int, optional): How many study examples should be included? Defaults to 1.
        n_reshuffle (int, optional): How many times to reshuffle to get new query-study example pairs. Defaults to 50
    Returns:
        pd.DataFrame: Dataframe with columns: n_perm, alphabet, transformation, query, study.
    """
    # data set should be csv with (n_permutations, alphabet, study examples, query, problem type)
    n = 0
    # initialize data frame:
    cols = ["n_perm", "alphabet", "transformation", "problem", "query_length"] #+ ["study" + str(i) for i in range(1, n_study+1)]
    dataset = pd.DataFrame(columns=cols)
    for n_perm in tqdm(alphabet_permutations):
        alphabets_n_perm = []
        for _ in range(alphabets_per_permutation):
            while True:
                # permute alphabet:
                _, _, alphabet = k_derange(n_perm, list(string.ascii_lowercase))
                alph_string = " ".join(alphabet)
                if alph_string not in alphabets_n_perm:
                    alphabets_n_perm.append(alph_string)
                    break
            
                
            for func_id in transformations.keys():
                problems, query_lengths = [], []
                for length in prob_lengths:
                    for i in range(len(alphabet)-length+1):
                        try:
                            query, solution = transformations[func_id](alphabet[i:i+length+1], alphabet)
                            problem = " ".join(query) + " > " + " ".join(solution)
                            problems.append(problem)
                            query_lengths.append(len(query))
                            n += 1
                        except IndexError:
                            pass
                # generate study examples by reshuffling the problems n_reshuffle times:
                reshuffled_datasets = []
                study_examples = np.array([np.random.permutation(problems) for _ in range(n_study)])
                separator = " | "
                study_examples_joined = np.array([separator.join(study_examples[:, i]) for i in range(study_examples.shape[1])])
                for _ in range(n_reshuffle):
                    reshuffled_data = pd.DataFrame(
                        {"n_perm" : [n_perm for _ in range(len(problems))],
                        "alphabet" : [alph_string for _ in range(len(problems))],
                        "transformation" : [transformations[func_id].__name__ for _ in range(len(problems))],
                        "study" : study_examples_joined,
                        "problem" : problems,
                        "query_length" : query_lengths
                        })
                    reshuffled_datasets.append(reshuffled_data)
                dataset = pd.concat([dataset] + reshuffled_datasets, ignore_index=True)
            if n_perm == 1:
                break
    
    print(f"Generated {n:,} unique queries.")

    # drop rows where problem is part of the study examples:
    dataset = dataset[~dataset.apply(lambda row: row["problem"] in row["study"], axis=1)]
    # filter out problems that are in Lewis & Mitchell dataset:
    l_m_problems = get_unique_problems_counterfactual_analogy()
    dataset = dataset[~dataset["problem"].isin(l_m_problems)]
    print(l_m_problems[:5])
    print(dataset["problem"].head())
    print(f"Resulting in {dataset.shape[0]:,} total samples.")
    return dataset


def dataset_to_disk(
        dataset: pd.DataFrame, 
        directory="data", 
        train_ratio: float=0.9
        ) -> None:
    
    # shuffle dataset:
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    rows = dataset.shape[0]
    max_train_id = round(train_ratio * rows)
    train = dataset.iloc[:max_train_id, :]

    # get lewis mitchell problems and remove them from training set:
    lewis_mitchell_problem_set = get_unique_problems_counterfactual_analogy()
    train[~train["problem"].isin(lewis_mitchell_problem_set)]

    # save datasets as csv:
    train.to_csv(f"{directory}/train.csv", index=False)
    val = dataset.iloc[max_train_id:, :]
    val.to_csv(f"{directory}/val.csv", index=False)
    print(f"Done. {train.shape[0]:,} training samples and {val.shape[0]:,} validation samples written to disk.")


ALL_TRANSFORMATIONS = {
    1 : extend_sequence,
    2 : succ,
    3 : pred,
    4 : remove_redundant,
    5 : fix_alphabetic_seq,
    6 : sort,
    7 : sort_group,
    8 : rr_interleave,
    9 : rr_succ,
    10 : fix_extend,
    11 : rr_sort,
    12 : extend_pred,
    13 : fix_interleave,
    14: extend_group,
    15: extend_extend_succ,
    16: fix_pred_succ,
    17 : reverse,
    18 : shift,
    19 : replicate,
}


def main():
    parser = argparse.ArgumentParser(
        description="""Overview of Transformations\n
        \n
         Transformations present in both data splits\n
        1.	Extend Sequence (a b c d -> a b c d e)\n
        2.	Successor (a b c d -> a b c e)\n
        3.	Predecessor (b c d e -> a c d e)\n
        4.	Remove Redundant letter (a b b c d -> a b c d)\n
        5.	Fix alphabetic sequence (a b c w e -> a b c d e)\n
        6.	Sort (a d c b e -> a b c d e)\n
        7.	Sort + Group (aa dd cc bb ee -> aa bb cc dd ee)\n
        8.	Remove redundant + Interleave (a x b x b x c x d -> a x b x c x d)\n
        9.	Remove Redundant + successor (a b b c d -> a b c e)\n
        10.	Fix alphabetic sequence + extend sequence (a b c w -> a b c d e)\n
        \n
        Transformations only present in validation/ test splits:\n
        11. Remove Redundant + Sort (a d d c b e -> a b c d e)\n
        12. Extend Sequence + Predecessor (b c d e -> a c d e f)\n
        13. Fix alphabetic sequence + Interleave (a f b f c f w f e -> a f b f c f d f e)\n
        14. Extend sequence + Group (aa bb cc dd -> aa bb cc dd ee)\n
        15. Extend Sequence + Extend Sequence + Successor (a b c d -> a b c d e g)\n
        16. Fix alphabetic Sequence + Predecessor + Successor (a b c w e -> a a c d f)\n
        17. Reverse (a b c d -> d c b a)\n
        18. Shift (a b c d -> e f g h)\n
        19. Replicate (a b c d -> a b c d a b c d)""",
    formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--transformations', default="all", type=str, help='Comma-separated list of integers of transformations to include (e.g., 1,2 will include extend sequence and successor). Defaults to all.')
    parser.add_argument('--data_dir', default="data", help='The directory in which the data set will be saved')
    parser.add_argument('--n_reshuffle', default=50, type=int, help='How many times to reshuffle the data to get new problem-study example pairs. Defaults to 50.')
    parser.add_argument('--alphabets_per_permutation', default=1, type=int, help='How many unique alphabets to generate per permutation level. Defaults to 1.')
    args = parser.parse_args()
    
    # set seed for reproducibility:
    np.random.seed(1)

    match args.transformations: 
        case "all":
            transformations = ALL_TRANSFORMATIONS
        case "base":
            transformations = {id: ALL_TRANSFORMATIONS[id] for id in range(1, 7)}
        case _:
            func_ids = [int(id) for id in args.transformations.split(',')]
            transformations = {id: ALL_TRANSFORMATIONS[id] for id in func_ids}
    
    dataset = generate_dataset(
        transformations=transformations, 
        n_reshuffle=args.n_reshuffle,
        alphabets_per_permutation=args.alphabets_per_permutation
    )
    # create directory if it doesnt exist yet
    # if not os.path.exists(args.data_dir):
    #     os.makedirs(args.data_dir)
    #     print(f"Created '{args.data_dir}' directory.")
    # dataset_to_disk(dataset, directory=args.data_dir)


if __name__ == "__main__":
    main()

