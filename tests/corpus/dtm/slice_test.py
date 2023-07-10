from typing import Callable

import numpy as np
import pandas as pd
import pytest
import scipy

from penelope.corpus import VectorizedCorpus
from penelope.corpus.token2id import id2token2token2id

from ...utils import create_abc_corpus, simple_vectorized_abc_corpus

# pylint: disable=redefined-outer-name


@pytest.fixture
def slice_corpus() -> VectorizedCorpus:
    return simple_vectorized_abc_corpus()


def test_slice_by_indices():
    make_corpus: Callable[[], VectorizedCorpus] = lambda: create_abc_corpus(
        [
            [2, 1, 4, 1],
            [2, 2, 3, 0],
            [2, 3, 2, 0],
        ]
    )

    corpus: VectorizedCorpus = make_corpus()
    sliced_corpus: VectorizedCorpus = corpus.slice_by_indices([])
    assert sliced_corpus.data.shape[1] == 0

    corpus = make_corpus()
    sliced_corpus: VectorizedCorpus = corpus.slice_by_indices(None)
    assert sliced_corpus.data.shape[1] == 0

    corpus = make_corpus()
    sliced_corpus: VectorizedCorpus = corpus.slice_by_indices([0, 2])
    assert sliced_corpus is not corpus
    assert (sliced_corpus.data.todense() == [[2, 4], [2, 3], [2, 2]]).all().all()

    corpus = make_corpus()
    sliced_corpus: VectorizedCorpus = corpus.slice_by_indices([0, 2], inplace=True)
    assert sliced_corpus is corpus
    assert (sliced_corpus.data.todense() == [[2, 4], [2, 3], [2, 2]]).all().all()


def test_normalize_with_default_arguments_returns_matrix_normalized_by_l1_norm_for_each_row():
    bag_term_matrix = np.array([[4, 3, 7, 1], [6, 7, 4, 2]])
    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    df = pd.DataFrame({'year': [2013, 2014]})
    v_corpus: VectorizedCorpus = VectorizedCorpus(bag_term_matrix, token2id=token2id, document_index=df)
    n_corpus = v_corpus.normalize()
    E = np.array([[4, 3, 7, 1], [6, 7, 4, 2]]) / (np.array([[15, 19]]).T)
    assert (E == n_corpus.bag_term_matrix).all()


def test_normalize_with_keep_magnitude():
    bag_term_matrix = np.array([[4, 3, 7, 1], [6, 7, 4, 2]])
    bag_term_matrix = scipy.sparse.csr_matrix(bag_term_matrix)

    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    df = pd.DataFrame({'year': [2013, 2014]})

    v_corpus: VectorizedCorpus = VectorizedCorpus(bag_term_matrix, token2id=token2id, document_index=df)
    n_corpus = v_corpus.normalize(keep_magnitude=True)

    factor = 15.0 / 19.0
    E = np.array([[4.0, 3.0, 7.0, 1.0], [6.0 * factor, 7.0 * factor, 4.0 * factor, 2.0 * factor]])
    assert np.allclose(E, n_corpus.bag_term_matrix.todense())


def test_slice_by_indices2():
    # self: IVectorizedCorpusProtocol, n_top: int, sort_indices: bool=False, override: bool=False) -> np.ndarray:
    corpus: VectorizedCorpus = create_abc_corpus(
        [
            [4, 1, 3, 3, 1],
            [3, 2, 0, 1, 0],
            [2, 3, 2, 3, 0],
        ]
    )
    corpus._overridden_term_frequency = np.array([4, 5, 5, 3, 1])  # pylint: disable=protected-access

    sliced_corpus: VectorizedCorpus = corpus.slice_by_indices([1, 2, 4])

    assert (
        (
            sliced_corpus.data.todense()
            == (
                [
                    [1, 3, 1],
                    [2, 0, 0],
                    [3, 2, 0],
                ]
            )
        )
        .all()
        .all()
    )
    assert sliced_corpus.token2id == {'b': 0, 'c': 1, 'e': 2}
    assert sliced_corpus.overridden_term_frequency.tolist() == [5, 5, 1]
    assert sliced_corpus.term_frequency.tolist() == [6, 5, 1]
    assert sliced_corpus.term_frequency0.tolist() == [5, 5, 1]


def test_slice_by_tf(slice_corpus: VectorizedCorpus):
    corpus = slice_corpus.slice_by_tf(1)
    assert corpus.data.shape == slice_corpus.data.shape
    assert corpus.data.sum() == slice_corpus.data.sum()

    corpus = slice_corpus.slice_by_tf(4)
    assert corpus.term_frequency.tolist() == [10, 10, 11]

    corpus = slice_corpus.slice_by_tf(10)
    assert corpus.term_frequency.tolist() == [10, 10, 11]

    corpus = slice_corpus.slice_by_tf(12)
    assert corpus.term_frequency.tolist() == []


def test_slice_by_tf_when_exists_tokens_below_count_returns_filtered_corpus(slice_corpus: VectorizedCorpus):
    # Act
    t_corpus: VectorizedCorpus = slice_corpus.slice_by_tf(6)

    # Assert
    expected_bag_term_matrix = np.array([[2, 1, 4], [2, 2, 3], [2, 3, 2], [2, 4, 1], [2, 0, 1]])

    assert {'a': 0, 'b': 1, 'c': 2} == t_corpus.token2id
    assert [10, 10, 11] == t_corpus.term_frequency.tolist()
    assert (expected_bag_term_matrix == t_corpus.bag_term_matrix).all()


def test_slice_by_tf_when_all_below_below_n_count_returns_empty_corpus(slice_corpus: VectorizedCorpus):
    t_corpus: VectorizedCorpus = slice_corpus.slice_by_tf(20)

    assert {} == t_corpus.token2id
    assert [] == t_corpus.term_frequency.tolist()
    assert (np.empty((5, 0)) == t_corpus.bag_term_matrix).all()


def test_slice_by_tf_when_all_tokens_above_n_count_returns_same_corpus(slice_corpus: VectorizedCorpus):
    t_corpus = slice_corpus.slice_by_tf(1)

    assert slice_corpus.token2id == t_corpus.token2id
    assert (slice_corpus.term_frequency == t_corpus.term_frequency).all()
    assert np.allclose(slice_corpus.bag_term_matrix.todense().A, t_corpus.bag_term_matrix.todense().A)


def test_slice_by_n_top_when_all_tokens_above_n_count_returns_same_corpus(slice_corpus: VectorizedCorpus):
    t_corpus = slice_corpus.slice_by_n_top(4)

    assert slice_corpus.token2id == t_corpus.token2id
    assert (slice_corpus.term_frequency == t_corpus.term_frequency).all()
    assert np.allclose(slice_corpus.bag_term_matrix.todense().A, t_corpus.bag_term_matrix.todense().A)


def test_slice_by_n_top_when_n_top_less_than_n_tokens_returns_corpus_with_top_n_counts(slice_corpus: VectorizedCorpus):
    t_corpus = slice_corpus.slice_by_n_top(2)

    expected_bag_term_matrix = np.array([[2, 4], [2, 3], [2, 2], [2, 1], [2, 1]])

    assert {'a': 0, 'c': 1} == t_corpus.token2id
    assert t_corpus.term_frequency.tolist() == [10, 11]
    assert (expected_bag_term_matrix == t_corpus.bag_term_matrix).all()


def test_term_frequencies(slice_corpus: VectorizedCorpus):
    assert slice_corpus.term_frequency.tolist() == [10, 10, 11, 3]


def test_compress():
    corpus: VectorizedCorpus = create_abc_corpus(
        [
            [2, 1, 4, 1],
            [2, 2, 3, 0],
            [2, 3, 2, 0],
        ]
    )

    compressed_corpus, _, _ = corpus.compress()
    assert (corpus.data.todense() == compressed_corpus.data.todense()).all().all()
    assert compressed_corpus is corpus

    compressed_corpus, _, _ = corpus.compress(inplace=True)
    assert (corpus.data.todense() == compressed_corpus.data.todense()).all().all()
    assert compressed_corpus is corpus

    corpus: VectorizedCorpus = create_abc_corpus(
        [
            [0, 2, 1, 0, 4, 1, 0],
            [0, 2, 2, 0, 3, 0, 0],
            [0, 2, 3, 0, 2, 0, 0],
        ]
    )
    compressed_corpus, m, indices = corpus.compress(inplace=False)
    assert (
        (
            compressed_corpus.data.todense()
            == [
                [2, 1, 4, 1],
                [2, 2, 3, 0],
                [2, 3, 2, 0],
            ]
        )
        .all()
        .all()
    )

    assert compressed_corpus.token2id == {w: i for i, w in enumerate(['b', 'c', 'e', 'f'])}
    assert (indices == [1, 2, 4, 5]).all()
    assert m == {j: i for i, j in enumerate(indices)}

    inv_m = {new_id: old_id for old_id, new_id in m.items()}
    assert compressed_corpus.term_frequency.tolist() == [
        corpus.term_frequency[inv_m[new_id]] for new_id in sorted(inv_m)
    ]

    compressed_corpus, m, indices = corpus.compress(inplace=False, extra_keep_ids=[0, 6])
    assert (indices == [0, 1, 2, 4, 5, 6]).all()
    assert (
        (
            compressed_corpus.data.todense()
            == [
                [0, 2, 1, 4, 1, 0],
                [0, 2, 2, 3, 0, 0],
                [0, 2, 3, 2, 0, 0],
            ]
        )
        .all()
        .all()
    )


def test_nlargest():
    # self: IVectorizedCorpusProtocol, n_top: int, sort_indices: bool=False, override: bool=False) -> np.ndarray:
    corpus: VectorizedCorpus = create_abc_corpus(
        [
            [4, 1, 3, 3, 1],
            [3, 2, 0, 1, 0],
            [2, 3, 2, 3, 0],
        ]
    )
    assert corpus.term_frequency.tolist() == [9, 6, 5, 7, 1]
    assert corpus.overridden_term_frequency is None

    assert corpus.nlargest(2).tolist() == [3, 0]
    assert corpus.nlargest(2, sort_indices=False, override=False).tolist() == [3, 0]
    assert corpus.nlargest(2, sort_indices=False, override=True).tolist() == [3, 0]
    assert corpus.nlargest(3, sort_indices=True, override=True).tolist() == [0, 1, 3]

    corpus._overridden_term_frequency = np.array([4, 5, 5, 3, 1])  # pylint: disable=protected-access
    assert (corpus.overridden_term_frequency == np.array([4, 5, 5, 3, 1])).all()
    assert corpus.nlargest(3, sort_indices=True, override=True).tolist() == [0, 1, 2]


def test_where_is_above_threshold_with_keeps():
    values = np.array([0, 0, 2, 5, 1, 8, 4, 0])

    indices = VectorizedCorpus.where_is_above_threshold_with_keeps(values, threshold=4)
    assert indices.tolist() == [3, 5, 6]

    indices = VectorizedCorpus.where_is_above_threshold_with_keeps(values, threshold=4, keep_indices=[])
    assert indices.tolist() == [3, 5, 6]

    indices = VectorizedCorpus.where_is_above_threshold_with_keeps(values, threshold=4, keep_indices=[0, 1])
    assert indices.tolist() == [0, 1, 3, 5, 6]


def test_translate_to_vocab():
    A: VectorizedCorpus = create_abc_corpus(
        [
            # 0 1  2  3  4  5  6  7
            # x a  b  c  y  d  e  z
            [0, 0, 0, 3, 0, 0, 0, 0],
            [0, 0, 0, 5, 2, 0, 7, 4],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 2, 6, 0, 0, 3, 0],
        ],
        token2id=id2token2token2id({0: 'x', 1: 'a', 2: 'b', 3: 'c', 4: 'y', 5: 'd', 6: 'e', 7: 'z'}),
    )

    B: VectorizedCorpus = create_abc_corpus(
        [
            # 0 1  2  3  4  5  6  7  8
            # i c  g  f  a  d  b  h  e
            [0, 0, 0, 0, 3, 0, 0, 0, 4],
            [0, 3, 0, 4, 0, 0, 0, 7, 0],
            [0, 0, 4, 0, 9, 1, 0, 0, 0],
            [0, 2, 0, 0, 6, 0, 0, 3, 0],
        ],
        token2id=id2token2token2id({0: 'i', 1: 'c', 2: 'g', 3: 'f', 4: 'a', 5: 'd', 6: 'b', 7: 'h', 8: 'e'}),
    )
    """
    Translate B to A:
        expected common words: a b c d e
        expected target shape: A.shape
        expected old  indices: A.shape
        expected index translation:
            a: 4 => 1
            b: 6 => 2
            c: 1 => 3
            d: 5 => 5
            e: 8 => 6
    """
    expected_corpus = create_abc_corpus(
        [
            # x a  b  c  y  d  e  z
            [0, 3, 0, 0, 0, 0, 4, 0],
            [0, 0, 0, 3, 0, 0, 0, 0],
            [0, 9, 0, 0, 0, 1, 0, 0],
            [0, 6, 0, 2, 0, 0, 0, 0],
        ],
        token2id=A.token2id,
    )

    result_corpus = B.translate_to_vocab(A.id2token, inplace=False)

    assert result_corpus.shape == A.shape
    assert (result_corpus.data.todense() == expected_corpus.data.todense()).all().all()
