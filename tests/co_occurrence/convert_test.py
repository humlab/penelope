import os

import pandas as pd
import penelope.co_occurrence as co_occurrence
import penelope.co_occurrence.partition_by_document as co_occurrence_module
import penelope.corpus.dtm as dtm
import pytest
from penelope.corpus import DocumentIndexHelper, Token2Id
from penelope.type_alias import CoOccurrenceDataFrame, DocumentIndex
from tests.test_data.corpus_fixtures import SIMPLE_CORPUS_ABCDE_5DOCS
from tests.utils import very_simple_corpus

jj = os.path.join


def test_filename_to_folder_and_tag():

    filename = f'./tests/test_data/VENUS/VENUS{co_occurrence.FILENAME_POSTFIX}'
    folder, tag = co_occurrence.to_folder_and_tag(filename)
    assert folder == './tests/test_data/VENUS'
    assert tag == 'VENUS'


def test_folder_and_tag_to_filename():
    expected_filename = f'./tests/test_data/VENUS/VENUS{co_occurrence.FILENAME_POSTFIX}'
    folder = './tests/test_data/VENUS'
    tag = 'VENUS'
    filename = co_occurrence.to_filename(folder=folder, tag=tag)
    assert filename == expected_filename


@pytest.mark.parametrize(
    'filename', ['concept_data_co-occurrence.csv', f'concept_data{co_occurrence.FILENAME_POSTFIX}']
)
def test_load_co_occurrences(filename):

    filename = jj('./tests/test_data', filename)

    co_occurrences = co_occurrence.load_co_occurrences(filename)

    assert co_occurrences is not None
    assert 13 == len(co_occurrences)
    assert 18 == co_occurrences.value.sum()
    assert 3 == int(co_occurrences[(co_occurrences.w1 == 'g') & (co_occurrences.w2 == 'a')]['value'])
    assert (['w1', 'w2', 'value', 'value_n_d', 'value_n_t'] == co_occurrences.columns).all()


@pytest.mark.parametrize(
    'filename', ['concept_data_co-occurrence.csv', f'concept_data{co_occurrence.FILENAME_POSTFIX}']
)
def test_store_co_occurrences(filename):

    source_filename = jj('./tests/test_data', filename)
    target_filename = jj('./tests/output', filename)

    co_occurrences = co_occurrence.load_co_occurrences(source_filename)

    co_occurrence.store_co_occurrences(target_filename, co_occurrences)

    assert os.path.isfile(target_filename)

    co_occurrences = co_occurrence.load_co_occurrences(target_filename)
    assert co_occurrences is not None

    os.remove(target_filename)


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


def test_load_and_store_bundle():

    filename = co_occurrence.to_filename(folder='./tests/test_data/VENUS', tag='VENUS')

    bundle: co_occurrence.Bundle = co_occurrence.Bundle.load(filename)

    assert bundle is not None
    assert isinstance(bundle.corpus, dtm.VectorizedCorpus)
    assert isinstance(bundle.co_occurrences, pd.DataFrame)
    assert isinstance(bundle.compute_options, dict)
    assert bundle.folder == './tests/test_data/VENUS'
    assert bundle.tag == 'VENUS'

    os.makedirs('./tests/output/MARS', exist_ok=True)

    expected_filename = co_occurrence.to_filename(folder='./tests/output/MARS', tag='MARS')

    bundle.store(folder='./tests/output/MARS', tag='MARS')

    assert bundle.co_occurrence_filename == expected_filename
    assert os.path.isfile(bundle.co_occurrence_filename)


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
