import collections
from typing import Iterable

import numpy as np
from penelope.corpus import CorpusVectorizer, Token2Id, VectorizedCorpus
from penelope.pipeline.co_occurrence.tasks import CoOccurrencePayload, TTM_to_co_occurrence_DTM
from penelope.type_alias import DocumentIndex
from tests.fixtures import SIMPLE_CORPUS_ABCDE_5DOCS, very_simple_corpus


def test_co_occurrence_matrix_of_corpus_returns_correct_result():

    expected_token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
    expected_matrix = np.matrix([[0, 6, 4, 3, 3], [0, 0, 2, 1, 4], [0, 0, 0, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)

    v_corpus = CorpusVectorizer().fit_transform(corpus, already_tokenized=True, vocabulary=corpus.token2id)

    term_term_matrix = v_corpus.co_occurrence_matrix()

    assert (term_term_matrix.todense() == expected_matrix).all()
    assert expected_token2id == v_corpus.token2id


def test_TTM_to_COO_DTM_using_lil_matrix():

    t_corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)
    t_token2id = Token2Id(t_corpus.token2id)
    t_document_index: DocumentIndex = t_corpus.document_index

    stream: Iterable[CoOccurrencePayload] = (
        CoOccurrencePayload(
            document_id,
            CorpusVectorizer()
            .fit_transform([doc], already_tokenized=True, vocabulary=t_corpus.token2id)
            .co_occurrence_matrix(),
            collections.Counter(),
        )
        for document_id, doc in enumerate(t_corpus)
    )

    corpus: VectorizedCorpus = TTM_to_co_occurrence_DTM(stream, t_token2id, t_document_index)

    assert corpus is not None
