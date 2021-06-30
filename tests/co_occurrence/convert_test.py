import os

import penelope.co_occurrence as co_occurrence
import pytest
import scipy
from penelope.co_occurrence import term_term_matrix_to_co_occurrences, truncate_by_global_threshold
from penelope.co_occurrence.persistence import co_occurrence_filename, document_index_filename, vocabulary_filename
from penelope.corpus import DocumentIndexHelper, Token2Id, dtm
from penelope.corpus.dtm.ttm_legacy import LegacyCoOccurrenceMixIn
from penelope.type_alias import CoOccurrenceDataFrame, DocumentIndex
from tests.fixtures import SIMPLE_CORPUS_ABCDE_5DOCS, very_simple_corpus, very_simple_term_term_matrix

jj = os.path.join


def test_to_co_occurrence_matrix():

    text_corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)

    term_term_matrix1 = very_simple_term_term_matrix(text_corpus)

    term_term_matrix2 = co_occurrence.to_co_occurrence_matrix(text_corpus)

    assert (term_term_matrix1 != term_term_matrix2).nnz == 0


@pytest.mark.skip(reason="Deprecated co_occurrences_to_co_occurrence_corpus (or not tested)")
def test_co_occurrences_to_co_occurrence_corpus():

    folder, tag = './tests/test_data/ABCDEFG_7DOCS_CONCEPT', "ABCDEFG_7DOCS_CONCEPT"

    co_occurrences: CoOccurrenceDataFrame = co_occurrence.load_co_occurrences(co_occurrence_filename(folder, tag))
    document_index: DocumentIndex = DocumentIndexHelper.load(document_index_filename(folder, tag)).document_index
    token2id: Token2Id = Token2Id.load(vocabulary_filename(folder, tag))

    corpus = LegacyCoOccurrenceMixIn.from_co_occurrences(
        co_occurrences=co_occurrences,
        document_index=document_index,
        token2id=token2id,
    )

    assert corpus.data.sum() == co_occurrences.value.sum()
    assert corpus.data.shape[0] == len(document_index)
    assert corpus.data.shape[1] == len(co_occurrences[["w1_id", "w2_id"]].drop_duplicates())


def test_truncate_by_global_threshold():
    folder, tag = './tests/test_data/VENUS', "VENUS"
    co_occurrences: CoOccurrenceDataFrame = co_occurrence.load_co_occurrences(co_occurrence_filename(folder, tag))
    truncated_co_occurrences = truncate_by_global_threshold(co_occurrences=co_occurrences, threshold=2)
    assert len(truncated_co_occurrences) < len(co_occurrences) is not None
    # FIXME Add more tests/asserts


def test_term_term_matrix_to_co_occurrences_with_ignore_ids():

    text_corpus = very_simple_corpus(
        data=[
            ('tran_2019_01_test.txt', ['*', 'b', 'c', 'c']),
            ('tran_2019_02_test.txt', ['a', '*', '*', 'd']),
            ('tran_2019_03_test.txt', ['a', 'e', 'e', 'b']),
            ('tran_2020_01_test.txt', ['*', 'c', 'd', 'a']),
            ('tran_2020_02_test.txt', ['a', 'b', '*', '*']),
        ]
    )
    token2id: Token2Id = Token2Id(text_corpus.token2id)

    term_term_matrix = (
        dtm.CorpusVectorizer()
        .fit_transform(text_corpus, already_tokenized=True, vocabulary=text_corpus.token2id)
        .co_occurrence_matrix()
    )

    pad_id = token2id['*']

    co_occurrences = term_term_matrix_to_co_occurrences(
        term_term_matrix=term_term_matrix,
        threshold_count=1,
        ignore_ids=set([pad_id]),
    )

    assert not (co_occurrences.w1_id == pad_id).any()
    assert not (co_occurrences.w2_id == pad_id).any()


def test_term_term_matrix_to_co_occurrences_with_no_ignore_ids():

    text_corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)
    term_term_matrix: scipy.sparse.spmatrix = very_simple_term_term_matrix(text_corpus)

    co_occurrences = term_term_matrix_to_co_occurrences(
        term_term_matrix=term_term_matrix,
        threshold_count=1,
        ignore_ids=None,
    )

    fg = text_corpus.token2id.get
    assert co_occurrences.value.sum() == term_term_matrix.sum()
    assert 4 == int(co_occurrences[((co_occurrences.w1_id == fg('a')) & (co_occurrences.w2_id == fg('c')))].value)
    assert 1 == int(co_occurrences[((co_occurrences.w1_id == fg('b')) & (co_occurrences.w2_id == fg('d')))].value)
