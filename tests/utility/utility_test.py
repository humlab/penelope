import numpy as np
import pytest
import scipy
import scipy.sparse as sp

from penelope import corpus as pc
from penelope import utility
from penelope.corpus.transformer import TransformProperty


def test_utils():
    assert utility.remove_snake_case('x_y') == 'X Y'

    assert utility.isint("1")
    assert not utility.isint("W")

    assert utility.extend({'a': 1}, {'b': 2}, c=3) == {'a': 1, 'b': 2, 'c': 3}
    assert utility.uniquify([5, 3, 2, 2, 1, 5]) == [5, 3, 2, 1]
    assert not utility.is_platform_architecture('32bit')
    # assert utility.trunc_year_by(np.array([1968]), 5) == [1965]
    # assert (utility.normalize_values(np.array([1.,1.])) == [0.5,0.5]).all()
    assert (utility.normalize_array(np.array([1.0, 1.0])) == [0.5, 0.5]).all()
    assert utility.timestamp(None) is not None
    assert utility.slim_title('slim_title')
    assert utility.split(['_'], 'slim_title') == ['slim', 'title']
    # assert utility.ls_sorted('./tests/testa_data/*') != []
    assert utility.dict_of_key_values_inverted_to_dict_of_value_key({'a': [1, 2]}) == {1: 'a', 2: 'a'}
    assert utility.dict_to_list_of_tuples({'a': 1, 'b': 2}) == [('a', 1), ('b', 2)]


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

    # from itertools import chain

    # y = dict(chain(*[[(v, k) for v in l] for k, l in x.items()]))
    # assert y == {1: 'a', 2: 'a', 3: 'b', 4: 'b'}

    y = {value: key for key in x for value in x[key]}
    assert y == {1: 'a', 2: 'a', 3: 'b', 4: 'b'}


def test_normalize_sparse_matrix_by_vector():
    data = np.array([[2, 1, 4, 1], [2, 2, 3, 0], [0, 0, 0, 0]])
    csr = scipy.sparse.csr_matrix(data)
    normalized_matrix = utility.normalize_sparse_matrix_by_vector(csr)
    expected = np.nan_to_num(data / data.sum(axis=1, keepdims=True), copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    assert np.allclose(normalized_matrix.todense(), expected)


def test_multiple_replace():
    tokens = [chr(ord('a') + i) for i in range(0, 10)]
    tokens_str = ' '.join(tokens)

    replace_map = {
        "d e f": "d_e_f",
        "i j": "i_j",
    }

    assert utility.multiple_replace(tokens_str, replace_map) == "a b c d_e_f g h i_j"

    replace_map = {
        "d e": "x_y",
        "d e f": "d_e_f",
        "i j": "i_j",
    }

    assert utility.multiple_replace(tokens_str, replace_map) == "a b c d_e_f g h i_j"


def test_csr2bow():
    A = np.array(
        [
            [0, 0, 0, 0],
            [5, 8, 0, 0],
            [0, 0, 3, 0],
            [0, 6, 0, 0],
        ]
    )
    M = sp.csr_matrix(A)

    BOW = list(pc.csr2bow(M))

    assert BOW == [[], [(0, 5), (1, 8)], [(2, 3)], [(1, 6)]]


def test_comma_str():
    C = utility.CommaStr
    assert C('a').add('a') == 'a' == C('a') + 'a'
    assert C('').add('a') == 'a' == C('') + 'a'
    assert C('a').add('b') == 'a,b' == C('a') + 'b'
    assert C('a,b').add('c') == 'a,b,c' == C('a,b') + 'c'
    assert C('a,b,c').add('b') == 'a,b,c' == C('a,b,c') + 'b'

    assert C('a,b,c').remove('b') == 'a,c' == C('a,b,c') - 'b'
    assert C('a,b,c').remove('a') == 'b,c' == C('a,b,c') - 'a'
    assert C('a,b,c').remove('c') == 'a,b' == C('a,b,c') - 'c'
    assert C('').remove('a') == '' == C('') - 'a'
    assert C('a,b,c').remove('d') == 'a,b,c' == C('a,b,c') - 'd'
    assert C('a,b,c').remove('a,b,c,d') == '' == C('a,b,c') - 'a,b,c,d'

    assert C('a,b,c') & C('a,b,c') == 'a,b,c'
    assert C('a,b,c') & C('a,b,d') == 'a,b'
    assert C('a,b,c') | C('a,b,d') == 'a,b,c,d'

    assert C('a,b,c').add('d').remove('d') == 'a,b,c' == C('a,b,c').add('d') - 'd'


def test_remove_all():
    C = utility.CommaStr
    assert C('a?1,b,c').remove('a') == 'b,c'
    assert C('a?1,b,c').remove('a?1') == 'b,c'
    assert C('a,b,c').remove('a?1') == 'b,c'
    assert C('a,a?1,a?b=2,b,c').remove('a?1') == 'b,c'
    # == (C('a,b,c') - 'a?1')


def test_find_keys():
    C = utility.CommaStr
    assert list(C('a,b,c').find_keys('a?1')) == ['a']
    assert list(C('a,b,c').find_keys('a')) == ['a']
    assert list(C('a?1,b,c').find_keys('a?1')) == ['a?1']
    assert list(C('a?1,b,c').find_keys('a')) == ['a?1']
    assert list(C('a?1,a?2,a,b,c').find_keys('a')) == ['a?1', 'a?2', 'a']

