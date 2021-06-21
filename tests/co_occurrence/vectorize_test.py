from typing import Iterable

import numpy as np
from penelope.co_occurrence.vectorize import VectorizedTTM, VectorizeType
from penelope.corpus import CorpusVectorizer, Token2Id, VectorizedCorpus
from penelope.pipeline.co_occurrence.tasks import CoOccurrenceCorpusBuilder, CoOccurrencePayload
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


def test_TTM_to_co_occurrence_DTM_using_LIL_matrix():

    source_corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)
    single_vocabulary = Token2Id(source_corpus.token2id)
    document_index: DocumentIndex = source_corpus.document_index

    stream: Iterable[CoOccurrencePayload] = (
        CoOccurrencePayload(
            document_id,
            vectorized_data={
                VectorizeType.Normal: VectorizedTTM(
                    vectorize_type=VectorizeType.Normal,
                    term_term_matrix=CorpusVectorizer()
                    .fit_transform([doc], already_tokenized=True, vocabulary=single_vocabulary.data)
                    .co_occurrence_matrix(),
                    term_window_counts={},
                    document_id=document_id,
                )
            },
        )
        for document_id, doc in enumerate(source_corpus)
    )

    pair_vocabulary: Token2Id = Token2Id()

    builder: CoOccurrenceCorpusBuilder = CoOccurrenceCorpusBuilder(
        vectorize_type=VectorizeType.Normal,
        document_index=document_index,
        pair_vocabulary=pair_vocabulary,
        single_vocabulary=single_vocabulary,
    )

    for payload in stream:
        builder.ingest_tokens(payload).add(payload)

    corpus: VectorizedCorpus = builder.corpus

    assert corpus is not None
