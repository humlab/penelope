import numpy as np
import pandas as pd
import penelope.corpus.dtm as dtm
import pytest
import scipy

from .utils import create_vectorized_corpus

# pylint: disable=redefined-outer-name


@pytest.fixture
def slice_corpus() -> dtm.VectorizedCorpus:
    return create_vectorized_corpus()


def test_normalize_with_default_arguments_returns_matrix_normalized_by_l1_norm_for_each_row():
    bag_term_matrix = np.array([[4, 3, 7, 1], [6, 7, 4, 2]])
    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    df = pd.DataFrame({'year': [2013, 2014]})
    v_corpus: dtm.VectorizedCorpus = dtm.VectorizedCorpus(bag_term_matrix, token2id, df)
    n_corpus = v_corpus.normalize()
    E = np.array([[4, 3, 7, 1], [6, 7, 4, 2]]) / (np.array([[15, 19]]).T)
    assert (E == n_corpus.bag_term_matrix).all()


def test_normalize_with_keep_magnitude():
    bag_term_matrix = np.array([[4, 3, 7, 1], [6, 7, 4, 2]])
    bag_term_matrix = scipy.sparse.csr_matrix(bag_term_matrix)

    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    df = pd.DataFrame({'year': [2013, 2014]})

    v_corpus: dtm.VectorizedCorpus = dtm.VectorizedCorpus(bag_term_matrix, token2id, df)
    n_corpus = v_corpus.normalize(keep_magnitude=True)

    factor = 15.0 / 19.0
    E = np.array([[4.0, 3.0, 7.0, 1.0], [6.0 * factor, 7.0 * factor, 4.0 * factor, 2.0 * factor]])
    assert np.allclose(E, n_corpus.bag_term_matrix.todense())


def test_slice_by_n_count_when_exists_tokens_below_count_returns_filtered_corpus(slice_corpus):

    # Act
    t_corpus: dtm.VectorizedCorpus = slice_corpus.slice_by_n_count(6)

    # Assert
    expected_bag_term_matrix = np.array([[2, 1, 4], [2, 2, 3], [2, 3, 2], [2, 4, 1], [2, 0, 1]])

    assert {'a': 0, 'b': 1, 'c': 2} == t_corpus.token2id
    assert {'a': 10, 'b': 10, 'c': 11} == t_corpus.term_frequency_mapping
    assert (expected_bag_term_matrix == t_corpus.bag_term_matrix).all()


def test_slice_by_n_count_when_all_below_below_n_count_returns_empty_corpus(slice_corpus):

    t_corpus: dtm.VectorizedCorpus = slice_corpus.slice_by_n_count(20)

    assert {} == t_corpus.token2id
    assert {} == t_corpus.term_frequency_mapping
    assert (np.empty((5, 0)) == t_corpus.bag_term_matrix).all()


def test_slice_by_n_count_when_all_tokens_above_n_count_returns_same_corpus(slice_corpus):

    t_corpus = slice_corpus.slice_by_n_count(1)

    assert slice_corpus.token2id == t_corpus.token2id
    assert slice_corpus.term_frequency_mapping == t_corpus.term_frequency_mapping
    assert np.allclose(slice_corpus.bag_term_matrix.todense().A, t_corpus.bag_term_matrix.todense().A)


def test_slice_by_n_top_when_all_tokens_above_n_count_returns_same_corpus(slice_corpus):

    t_corpus = slice_corpus.slice_by_n_top(4)

    assert slice_corpus.token2id == t_corpus.token2id
    assert slice_corpus.term_frequency_mapping == t_corpus.term_frequency_mapping
    assert np.allclose(slice_corpus.bag_term_matrix.todense().A, t_corpus.bag_term_matrix.todense().A)


def test_slice_by_n_top_when_n_top_less_than_n_tokens_returns_corpus_with_top_n_counts(slice_corpus):

    t_corpus = slice_corpus.slice_by_n_top(2)

    expected_bag_term_matrix = np.array([[2, 4], [2, 3], [2, 2], [2, 1], [2, 1]])

    assert {'a': 0, 'c': 1} == t_corpus.token2id
    assert {'a': 10, 'c': 11} == t_corpus.term_frequency_mapping
    assert (expected_bag_term_matrix == t_corpus.bag_term_matrix).all()


def test_term_frequencies(slice_corpus):
    assert slice_corpus.term_frequencies.tolist() == [10, 10, 11, 3]


def test_slice_by_term_frequency(slice_corpus):

    corpus = slice_corpus.slice_by_term_frequency(1)
    assert corpus.data.shape == slice_corpus.data.shape
    assert corpus.data.sum() == slice_corpus.data.sum()

    corpus = slice_corpus.slice_by_term_frequency(4)
    assert corpus.term_frequencies.tolist() == [10, 10, 11]

    corpus = slice_corpus.slice_by_term_frequency(10)
    assert corpus.term_frequencies.tolist() == [10, 10, 11]

    corpus = slice_corpus.slice_by_term_frequency(12)
    assert corpus.term_frequencies.tolist() == []
