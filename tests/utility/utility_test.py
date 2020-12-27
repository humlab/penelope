import numpy as np
import penelope.utility as utility
import pytest
import scipy


@pytest.mark.parametrize(
    'lst1, lst2, key, expected',
    [
        ([{'b': 2}], [{'c': 3}], 'a', [{'b': 2}]),
        ([{'a': 1, 'b': 2}], [{'a': 1, 'c': 3}], 'a', [{'a': 1, 'b': 2, 'c': 3}]),
        ([{'a': 1, 'b': 2}], [{'c': 3}], 'a', [{'a': 1, 'b': 2}]),
    ],
)
def test_lists_of_dicts_merged_by_key_when_results_are_expected(lst1, lst2, key, expected):

    result = utility.lists_of_dicts_merged_by_key(lst1, lst2, key=key)
    assert expected == result


@pytest.mark.parametrize(
    'lst1, lst2, key',
    [
        ([{'b': 2}], [{'a': 1, 'c': 3}], 'a'),
    ],
)
def test_lists_of_dicts_merged_by_key_when_exception_is_expected(lst1, lst2, key):

    with pytest.raises(ValueError):
        _ = utility.lists_of_dicts_merged_by_key(lst1, lst2, key=key)


def test_dict_of_lists_to_list_of_dicts():
    dl = {'a': [1, 2], 'b': [3, 4]}
    ld = utility.dict_of_lists_to_list_of_dicts(dl)
    assert ld == [{'a': 1, 'b': 3}, {'a': 2, 'b': 4}]


def test_dict_to_list_of_tuples():
    x = {'a': [1, 2], 'b': [3, 4]}
    y = utility.dict_to_list_of_tuples(x)
    assert y == [('a', [1, 2]), ('b', [3, 4])]


def test_dict_of_key_values_to_dict_of_value_key():
    x = {'a': [1, 2], 'b': [3, 4]}
    y = {k: v for k, v in utility.flatten([[(v, k) for v in l] for k, l in x.items()])}
    assert y == {1: 'a', 2: 'a', 3: 'b', 4: 'b'}

    from itertools import chain

    y = dict(chain(*[[(v, k) for v in l] for k, l in x.items()]))
    assert y == {1: 'a', 2: 'a', 3: 'b', 4: 'b'}

    y = {value: key for key in x for value in x[key]}
    assert y == {1: 'a', 2: 'a', 3: 'b', 4: 'b'}


def test_normalize_sparse_matrix_by_vector():

    data = np.array([[2, 1, 4, 1], [2, 2, 3, 0], [0, 0, 0, 0]])
    csr = scipy.sparse.csr_matrix(data)
    normalized_matrix = utility.normalize_sparse_matrix_by_vector(csr)
    expected = np.nan_to_num(data / data.sum(axis=1, keepdims=True), copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    assert np.allclose(normalized_matrix.todense(), expected)
