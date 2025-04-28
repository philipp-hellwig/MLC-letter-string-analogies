import argparse
from copy import deepcopy
import random
import string

import numpy as np
import pandas as pd


# TODO: Should we let predecessor problems spill over? i.e., a b c d -> z b c d
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
    if idx_first == (len(alphabet)-1):
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
def shift(prob_letters, *args):
    idx_last = alphabet.index(prob_letters[-1])
    if idx_last == (len(alphabet)-1-len(prob_letters)):
        raise IndexError("Sequence is too close to the end of the alphabet to be shifted")
    return [prob_letters, alphabet[idx_last+1:idx_last+1+len(prob_letters)]]


# 19. Replicate (a b c d -> a b c d a b c d)
def replicate(prob_letters, *args):
    return [prob_letters, list(np.tile(prob_letters, 2))]

# permute alphabet
def k_derange(k, alphabet):
    if k == 1:
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


ALL_TRANSFORMATIONS = {
    'extend_seq' : extend_sequence,
    'succ' : succ,
    'pred' : pred,
    'remove_redundant' : remove_redundant,
    'fix_alphabet' : fix_alphabetic_seq,
    'sort' : sort,
    "sort_group" : sort_group,
    "rr_interleave" : rr_interleave,
    "rr_succ" : rr_succ,
    "fix_extend" : fix_extend,
    "rr_sort" : rr_sort,
    "extend_pred" : extend_pred,
    "fix_interleave" : fix_interleave,
    "extend_group": extend_group,
    "extend_extend_succ": extend_extend_succ,
    "fix_pred_succ": fix_pred_succ,
    "reverse" : reverse,
    "shift" : shift,
    "replicate" : replicate,
}


def generate_dataset(
        alphabet_permutations: list=[1, 2, 5, 10, 20], 
        prob_lengths: list=[2,3,4,5,6],
        transformations=ALL_TRANSFORMATIONS,
        n_reshuffle: int=50
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
        # permute alphabet:
        _, _, alphabet = k_derange(n_perm, list(string.ascii_lowercase))
        alph_string = " ".join(alphabet)
        
        for name in transformations.keys():
            queries = []
            for length in prob_lengths:
                for i in range(len(alphabet)-length+1):
                    query = transformations[name](alphabet[i:i+length+1], alphabet)
                    query = " ".join(query[0]) + " > " + " ".join(query[1])
                    queries.append(query)
                    n += 1

            # generate study examples by reshuffling the queries n_reshuffle times:
            for _ in range(n_reshuffle):
                new_data = pd.DataFrame(
                    {"n_perm" : [n_perm for _ in range(len(queries))],
                    "alphabet" : [alph_string for _ in range(len(queries))],
                    "transformation" : [name for _ in range(len(queries))],
                    "study" : np.random.permutation(queries),
                    "query" : queries
                    })
                dataset = pd.concat([dataset, new_data], ignore_index=True)
    print(f"Generated {n} unique queries.")
    # drop rows where query is the same as study example:
    dataset = dataset.loc[dataset["query"] != dataset["study"], :]
    print(f"Resulting in {dataset.shape[0]} total samples.")
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
    train.to_csv(f"{directory}/train.csv", index=False)
    val = dataset.iloc[max_train_id:, :]
    val.to_csv(f"{directory}/val.csv", index=False)
    print(f"Done. {max_train_id} training samples and {rows - max_train_id} validation samples written to disk.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--illustrate', default=False, help='Instead of generating data, illustrate each letter string analogy with an example.')
    args = parser.parse_args()
    
    # set seed for reproducibility:
    np.random.seed(1)

    if args.illustrate:
        alphabet = list(string.ascii_lowercase)
        problem_letters = alphabet[2:6]
        print("Example analogy problems with alphabet:")
        print(alphabet, "\n")
        for i, trans in enumerate(ALL_TRANSFORMATIONS, start=1):
            print(f"{i}. '{trans}':")
            problem = ALL_TRANSFORMATIONS[trans](problem_letters, alphabet)
            print(problem, "\n")
    else:
        dataset = generate_dataset(n_reshuffle=50)
        dataset_to_disk(dataset)
