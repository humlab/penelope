import os
from tests.utils import generate_token2id, very_simple_corpus

import numpy as np

from penelope.co_occurrence import WindowsCorpus, to_co_occurrence_matrix, to_dataframe
from penelope.corpus import CorpusVectorizer
from .test_data.corpus_fixtures import SAMPLE_WINDOW_STREAM, SIMPLE_CORPUS_ABCDE_5DOCS

jj = os.path.join


def test_co_occurrence_matrix_of_corpus_returns_correct_result():

    expected_token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
    expected_matrix = np.matrix([[0, 6, 4, 3, 3], [0, 0, 2, 1, 4], [0, 0, 0, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)

    v_corpus = CorpusVectorizer().fit_transform(corpus, vocabulary=corpus.token2id)

    term_term_matrix = v_corpus.co_occurrence_matrix()

    assert (term_term_matrix.todense() == expected_matrix).all()
    assert expected_token2id == v_corpus.token2id


def test_to_dataframe_has_same_values_as_coocurrence_matrix():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)

    term_term_matrix = CorpusVectorizer().fit_transform(corpus, vocabulary=corpus.token2id).co_occurrence_matrix()

    df_coo = to_dataframe(
        term_term_matrix=term_term_matrix, id2token=corpus.id2token, documents=corpus.documents, n_count_threshold=1
    )

    assert df_coo.value.sum() == term_term_matrix.sum()
    assert 4 == int(df_coo[((df_coo.w1 == 'a') & (df_coo.w2 == 'c'))].value)
    assert 1 == int(df_coo[((df_coo.w1 == 'b') & (df_coo.w2 == 'd'))].value)


def test_to_coocurrence_matrix_yields_same_values_as_coocurrence_matrix():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)

    term_term_matrix1 = CorpusVectorizer().fit_transform(corpus, vocabulary=corpus.token2id).co_occurrence_matrix()

    term_term_matrix2 = to_co_occurrence_matrix(corpus)

    assert (term_term_matrix1 != term_term_matrix2).nnz == 0


def test_co_occurrence_given_windows_and_vocabulary_succeeds():

    vocabulary = generate_token2id([x[2] for x in SAMPLE_WINDOW_STREAM])

    windows_corpus = WindowsCorpus(SAMPLE_WINDOW_STREAM, vocabulary=vocabulary)

    v_corpus = CorpusVectorizer().fit_transform(windows_corpus, vocabulary=vocabulary)

    coo_matrix = v_corpus.co_occurrence_matrix()

    assert 10 == coo_matrix.todense()[vocabulary['b'], vocabulary['a']]
    assert 1 == coo_matrix.todense()[vocabulary['d'], vocabulary['c']]
