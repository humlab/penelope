import os

import numpy as np
from penelope.co_occurrence import WindowsCorpus
from penelope.corpus import CorpusVectorizer
from tests.utils import generate_token2id, very_simple_corpus

from ..test_data.corpus_fixtures import SAMPLE_WINDOW_STREAM, SIMPLE_CORPUS_ABCDE_5DOCS

jj = os.path.join


def test_co_occurrence_matrix_of_corpus_returns_correct_result():

    expected_token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
    expected_matrix = np.matrix([[0, 6, 4, 3, 3], [0, 0, 2, 1, 4], [0, 0, 0, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)

    v_corpus = CorpusVectorizer().fit_transform(corpus, already_tokenized=True, vocabulary=corpus.token2id)

    term_term_matrix = v_corpus.co_occurrence_matrix()

    assert (term_term_matrix.todense() == expected_matrix).all()
    assert expected_token2id == v_corpus.token2id


def test_co_occurrence_given_windows_and_vocabulary_succeeds():

    vocabulary = generate_token2id([x[2] for x in SAMPLE_WINDOW_STREAM])

    windows_corpus = WindowsCorpus(SAMPLE_WINDOW_STREAM, vocabulary=vocabulary)

    v_corpus = CorpusVectorizer().fit_transform(windows_corpus, already_tokenized=True, vocabulary=vocabulary)

    coo_matrix = v_corpus.co_occurrence_matrix()

    assert 10 == coo_matrix.todense()[vocabulary['b'], vocabulary['a']]
    assert 1 == coo_matrix.todense()[vocabulary['d'], vocabulary['c']]
