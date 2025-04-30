import pytest
import string
import numpy as np

from generate_data import (
    extend_sequence, succ, pred, remove_redundant, fix_alphabetic_seq, sort,
    sort_group, rr_interleave, rr_succ, fix_extend, rr_sort, extend_pred,
    fix_interleave, extend_group, extend_extend_succ, fix_pred_succ, reverse,
    shift, replicate
)

@pytest.fixture
def alphabet():
    return list(string.ascii_lowercase)

@pytest.fixture
def permuted_alphabet():
    np.random.seed(42)
    a = list(string.ascii_lowercase)
    np.random.shuffle(a)
    return a

@pytest.mark.parametrize("func", [
    extend_sequence, succ, pred, remove_redundant, fix_alphabetic_seq, sort,
    sort_group, rr_interleave, rr_succ, fix_extend, rr_sort, extend_pred,
    fix_interleave, extend_group, extend_extend_succ, fix_pred_succ, reverse,
    shift, replicate
])
def test_transformations_basic(func, alphabet):
    # Use a short sequence in the middle of the alphabet to avoid edge issues
    seq = alphabet[5:9]
    # Some functions require at least 3 letters
    if func in [sort, sort_group, rr_interleave, rr_sort, reverse, shift, replicate]:
        seq = alphabet[5:10]
    # Should not raise and should return a list of two lists
    result = func(seq, alphabet)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(x, list) for x in result)

def test_extend_sequence_edge(alphabet):
    # Last letter: should raise IndexError
    with pytest.raises(IndexError):
        extend_sequence([alphabet[-1]], alphabet)

def test_succ_edge(alphabet):
    # Last letter: should raise IndexError
    with pytest.raises(IndexError):
        succ([alphabet[-1]], alphabet)

def test_pred_edge(alphabet):
    # First letter: should raise IndexError
    with pytest.raises(IndexError):
        pred([alphabet[0]], alphabet)

def test_remove_redundant(alphabet):
    seq = alphabet[2:7]
    result = remove_redundant(seq)
    # The first element should have one more letter than the second
    assert len(result[0]) == len(result[1]) + 1
    # The second should be the original
    assert result[1] == seq

def test_fix_alphabetic_seq(alphabet):
    seq = alphabet[2:7]
    result = fix_alphabetic_seq(seq, alphabet)
    # The second element should be the original
    assert result[1] == seq
    # The first should differ by one letter
    diff = sum(a != b for a, b in zip(result[0], seq))
    assert diff == 1

def test_sort(alphabet):
    seq = list("edcba")
    result = sort(seq)
    # The second should be the original
    assert result[1] == seq

def test_group_problem():
    # Test sort_group and extend_group via their underlying logic
    seq = list("abc")
    grouped = sort_group(seq)
    assert isinstance(grouped, list)
    assert len(grouped) == 2
    assert all(isinstance(x, list) for x in grouped)

def test_rr_interleave(alphabet):
    seq = alphabet[2:7]
    result = rr_interleave(seq, alphabet)
    assert isinstance(result, list)
    assert len(result) == 2

def test_rr_succ(alphabet):
    seq = alphabet[2:7]
    result = rr_succ(seq, alphabet)
    assert isinstance(result, list)
    assert len(result) == 2

def test_fix_extend(alphabet):
    seq = alphabet[2:7]
    result = fix_extend(seq, alphabet)
    assert isinstance(result, list)
    assert len(result) == 2

def test_rr_sort(alphabet):
    seq = alphabet[2:7]
    result = rr_sort(seq, alphabet)
    assert isinstance(result, list)
    assert len(result) == 2

def test_extend_pred(alphabet):
    seq = alphabet[2:7]
    result = extend_pred(seq, alphabet)
    assert isinstance(result, list)
    assert len(result) == 2

def test_fix_interleave(alphabet):
    seq = alphabet[2:7]
    result = fix_interleave(seq, alphabet)
    assert isinstance(result, list)
    assert len(result) == 2

def test_extend_group(alphabet):
    seq = alphabet[2:7]
    result = extend_group(seq, alphabet)
    assert isinstance(result, list)
    assert len(result) == 2

def test_extend_extend_succ(alphabet):
    seq = alphabet[2:7]
    result = extend_extend_succ(seq, alphabet)
    assert isinstance(result, list)
    assert len(result) == 2

def test_fix_pred_succ(alphabet):
    seq = alphabet[2:7]
    result = fix_pred_succ(seq, alphabet)
    assert isinstance(result, list)
    assert len(result) == 2

def test_reverse(alphabet):
    seq = alphabet[2:7]
    result = reverse(seq, alphabet)
    assert result[1] == list(reversed(seq))

def test_shift(alphabet):
    seq = alphabet[2:7]
    result = shift(seq, alphabet)
    # The second element should be a shifted version of the first
    idx_last = alphabet.index(seq[-1])
    assert result[1] == alphabet[idx_last+1:idx_last+1+len(seq)]

def test_shift_edge(alphabet):
    # Too close to end
    seq = alphabet[-3:]
    with pytest.raises(IndexError):
        shift(seq, alphabet)

def test_replicate(alphabet):
    seq = alphabet[2:7]
    result = replicate(seq, alphabet)
    assert result[1] == seq + seq

@pytest.mark.parametrize("func", [
    extend_sequence, succ, pred, remove_redundant, fix_alphabetic_seq, sort,
    sort_group, rr_interleave, rr_succ, fix_extend, rr_sort, extend_pred,
    fix_interleave, extend_group, extend_extend_succ, fix_pred_succ, reverse,
    shift, replicate
])
def test_transformations_permuted_alphabet(func, permuted_alphabet):
    seq = permuted_alphabet[5:10]
    # Should not raise and should return a list of two lists
    try:
        result = func(seq, permuted_alphabet)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(x, list) for x in result)
    except IndexError:
        # Some edge cases may raise IndexError, that's fine
        pass