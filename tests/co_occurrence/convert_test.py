import os

import penelope.co_occurrence as co_occurrence
import penelope.co_occurrence.partition_by_document as co_occurrence_module
import penelope.corpus.dtm as dtm
import pytest
from penelope.corpus import DocumentIndexHelper, Token2Id
from penelope.type_alias import CoOccurrenceDataFrame, DocumentIndex
from tests.fixtures import SIMPLE_CORPUS_ABCDE_5DOCS, very_simple_corpus

jj = os.path.join


def test_to_vectorized_corpus():

    """Create an empty Bundle instance to get the filename right"""
    bundle: co_occurrence.Bundle = co_occurrence.Bundle(folder='./tests/test_data/VENUS', tag="VENUS")

    co_occurrences: CoOccurrenceDataFrame = co_occurrence.load_co_occurrences(bundle.co_occurrence_filename)
    document_index: DocumentIndex = DocumentIndexHelper.load(bundle.document_index_filename).document_index
    token2id: Token2Id = Token2Id.load(bundle.dictionary_filename)

    corpus = co_occurrence_module.co_occurrences_to_vectorized_corpus(
        co_occurrences=co_occurrences,
        document_index=document_index,
        token2id=token2id,
    )

    assert corpus.data.sum() == co_occurrences.value.sum()
    assert corpus.data.shape[0] == len(document_index)
    assert corpus.data.shape[1] == len(co_occurrences[["w1_id", "w2_id"]].drop_duplicates())


def test_to_co_occurrence_matrix():

    text_corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)

    term_term_matrix1 = (
        dtm.CorpusVectorizer()
        .fit_transform(text_corpus, already_tokenized=True, vocabulary=text_corpus.token2id)
        .co_occurrence_matrix()
    )

    term_term_matrix2 = co_occurrence.to_co_occurrence_matrix(text_corpus)

    assert (term_term_matrix1 != term_term_matrix2).nnz == 0


def test_to_dataframe_has_same_values_as_coocurrence_matrix():

    text_corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)

    term_term_matrix = (
        dtm.CorpusVectorizer()
        .fit_transform(text_corpus, already_tokenized=True, vocabulary=text_corpus.token2id)
        .co_occurrence_matrix()
    )

    co_occurrences = co_occurrence_module.term_term_matrix_to_co_occurrences(
        term_term_matrix=term_term_matrix,
        threshold_count=1,
        ignore_ids=None,
    )

    fg = text_corpus.token2id.get
    assert co_occurrences.value.sum() == term_term_matrix.sum()
    assert 4 == int(co_occurrences[((co_occurrences.w1_id == fg('a')) & (co_occurrences.w2_id == fg('c')))].value)
    assert 1 == int(co_occurrences[((co_occurrences.w1_id == fg('b')) & (co_occurrences.w2_id == fg('d')))].value)


def test_to_dataframe_coocurrence_matrix_with_paddings():

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

    co_occurrences = co_occurrence_module.term_term_matrix_to_co_occurrences(
        term_term_matrix=term_term_matrix,
        threshold_count=1,
        ignore_ids=set([pad_id]),
    )

    assert not (co_occurrences.w1_id == pad_id).any()
    assert not (co_occurrences.w2_id == pad_id).any()


def test_to_trends_data():
    filename: str = './tests/test_data/VENUS/VENUS_co-occurrence.csv.zip'
    bundle: co_occurrence.Bundle = co_occurrence.Bundle.load(filename, compute_corpus=False)

    trends_data = co_occurrence.to_trends_data(bundle).update()

    assert trends_data is not None


def test_load_options():
    filename: str = co_occurrence.to_filename(folder='./tests/test_data/VENUS', tag='VENUS')
    opts = co_occurrence.load_options(filename)
    assert opts is not None


@pytest.mark.skip(reason="not implemented")
def test_create_options_bundle():
    pass
