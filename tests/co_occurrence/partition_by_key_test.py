import os

import numpy as np
from penelope.co_occurrence import ContextOpts, CoOccurrenceComputeResult
from penelope.co_occurrence.partition_by_document import compute_corpus_co_occurrence
from penelope.corpus import CorpusVectorizer
from penelope.utility import dataframe_to_tuples
from tests.utils import very_simple_corpus

from ..test_data.corpus_fixtures import SIMPLE_CORPUS_ABCDE_5DOCS, SIMPLE_CORPUS_ABCDEFG_3DOCS

jj = os.path.join


def test_co_occurrence_matrix_of_corpus_returns_correct_result():

    expected_token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
    expected_matrix = np.matrix([[0, 6, 4, 3, 3], [0, 0, 2, 1, 4], [0, 0, 0, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)

    v_corpus = CorpusVectorizer().fit_transform(corpus, already_tokenized=True, vocabulary=corpus.token2id)

    term_term_matrix = v_corpus.co_occurrence_matrix()

    assert (term_term_matrix.todense() == expected_matrix).all()
    assert expected_token2id == v_corpus.token2id


def test_co_occurrence_without_no_concept_and_threshold_succeeds():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    expected_result = [('c', 'b', 2), ('b', 'g', 1), ('b', 'f', 1), ('g', 'f', 1)]

    value: CoOccurrenceComputeResult = compute_corpus_co_occurrence(
        stream=corpus,
        token2id=corpus.token2id,
        document_index=corpus.document_index,
        context_opts=ContextOpts(concept={'b'}, ignore_concept=False, context_width=1),
        global_threshold_count=0,
    )
    assert expected_result == dataframe_to_tuples(value.decoded_co_occurrences[['w1', 'w2', 'value']])


def test_co_occurrence_with_no_concept_succeeds():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)

    expected_result = {('d', 'a', 1), ('b', 'a', 1)}

    value: CoOccurrenceComputeResult = compute_corpus_co_occurrence(
        stream=corpus,
        token2id=corpus.token2id,
        document_index=corpus.document_index,
        context_opts=ContextOpts(concept={'g'}, ignore_concept=True, context_width=1),
        global_threshold_count=0,
    )
    assert expected_result == dataframe_to_tuples(value.decoded_co_occurrences[['w1', 'w2', 'value']])


def test_co_occurrence_with_thresholdt_succeeds():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    expected_result = {('g', 'a', 2)}

    value: CoOccurrenceComputeResult = compute_corpus_co_occurrence(
        stream=corpus,
        token2id=corpus.token2id,
        document_index=corpus.document_index,
        context_opts=ContextOpts(concept={'g'}, ignore_concept=False, context_width=1),
        global_threshold_count=2,
    )
    assert expected_result == dataframe_to_tuples(value.decoded_co_occurrences[['w1', 'w2', 'value']])
