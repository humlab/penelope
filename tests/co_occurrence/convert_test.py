import os

import pandas as pd
import penelope.co_occurrence as co_occurrence
import penelope.corpus.dtm as dtm
import pytest
from penelope.corpus import DocumentIndexHelper
from tests.test_data.corpus_fixtures import SIMPLE_CORPUS_ABCDE_5DOCS
from tests.utils import very_simple_corpus

jj = os.path.join


def test_filename_to_folder_and_tag():

    filename = f'./tests/test_data/VENUS/VENUS{co_occurrence.CO_OCCURRENCE_FILENAME_POSTFIX}'
    folder, tag = co_occurrence.filename_to_folder_and_tag(filename)
    assert folder == './tests/test_data/VENUS'
    assert tag == 'VENUS'


def test_folder_and_tag_to_filename():
    expected_filename = f'./tests/test_data/VENUS/VENUS{co_occurrence.CO_OCCURRENCE_FILENAME_POSTFIX}'
    folder = './tests/test_data/VENUS'
    tag = 'VENUS'
    filename = co_occurrence.folder_and_tag_to_filename(folder=folder, tag=tag)
    assert filename == expected_filename


@pytest.mark.parametrize(
    'filename', ['concept_data_co-occurrence.csv', f'concept_data{co_occurrence.CO_OCCURRENCE_FILENAME_POSTFIX}']
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
    'filename', ['concept_data_co-occurrence.csv', f'concept_data{co_occurrence.CO_OCCURRENCE_FILENAME_POSTFIX}']
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

    value_column = 'value'
    filename = co_occurrence.folder_and_tag_to_filename(folder='./tests/test_data/VENUS', tag='VENUS')

    index_filename = './tests/test_data/VENUS/VENUS_document_index.csv'

    co_occurrences = co_occurrence.load_co_occurrences(filename)
    document_index = DocumentIndexHelper.load(index_filename).document_index
    corpus = co_occurrence.to_vectorized_corpus(co_occurrences, document_index, value_column)

    assert corpus.data.shape[0] == len(document_index)
    assert corpus.data.shape[1] == len(co_occurrences.apply(lambda x: f"{x['w1']}/{x['w2']}", axis=1).unique())
    assert corpus is not None


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

    co_occurrences = co_occurrence.to_dataframe(
        term_term_matrix=term_term_matrix,
        id2token=text_corpus.id2token,
        document_index=text_corpus.document_index,
        threshold_count=1,
        ignore_pad=None,
    )

    assert co_occurrences.value.sum() == term_term_matrix.sum()
    assert 4 == int(co_occurrences[((co_occurrences.w1 == 'a') & (co_occurrences.w2 == 'c'))].value)
    assert 1 == int(co_occurrences[((co_occurrences.w1 == 'b') & (co_occurrences.w2 == 'd'))].value)


def test_to_dataframe_coocurrence_matrix_when_paddins():

    text_corpus = very_simple_corpus(
        data=[
            ('tran_2019_01_test.txt', ['*', 'b', 'c', 'c']),
            ('tran_2019_02_test.txt', ['a', '*', '*', 'd']),
            ('tran_2019_03_test.txt', ['a', 'e', 'e', 'b']),
            ('tran_2020_01_test.txt', ['*', 'c', 'd', 'a']),
            ('tran_2020_02_test.txt', ['a', 'b', '*', '*']),
        ]
    )

    term_term_matrix = (
        dtm.CorpusVectorizer()
        .fit_transform(text_corpus, already_tokenized=True, vocabulary=text_corpus.token2id)
        .co_occurrence_matrix()
    )

    co_occurrences = co_occurrence.to_dataframe(
        term_term_matrix=term_term_matrix,
        id2token=text_corpus.id2token,
        document_index=text_corpus.document_index,
        threshold_count=1,
        ignore_pad='*',
    )

    assert not (co_occurrences.w1 == '*').any()
    assert not (co_occurrences.w2 == '*').any()


def test_load_and_store_bundle():

    filename = co_occurrence.folder_and_tag_to_filename(folder='./tests/test_data/VENUS', tag='VENUS')

    bundle = co_occurrence.load_bundle(filename)

    assert bundle is not None
    assert isinstance(bundle.corpus, dtm.VectorizedCorpus)
    assert isinstance(bundle.co_occurrences, pd.DataFrame)
    assert isinstance(bundle.compute_options, dict)
    assert bundle.corpus_folder == './tests/test_data/VENUS'
    assert bundle.corpus_tag == 'VENUS'

    os.makedirs('./tests/output/MARS', exist_ok=True)
    filename = co_occurrence.folder_and_tag_to_filename(folder='./tests/output', tag='MARS')
    co_occurrence.store_bundle(filename, bundle)

    assert os.path.isfile(filename)


def test_to_trends_data():
    filename: str = './tests/test_data/VENUS/VENUS_co-occurrence.csv.zip'
    bundle = co_occurrence.load_bundle(filename, compute_corpus=False)

    trends_data = co_occurrence.to_trends_data(bundle).update()

    assert trends_data is not None


def test_load_options():
    filename: str = co_occurrence.folder_and_tag_to_filename(folder='./tests/test_data/VENUS', tag='VENUS')
    opts = co_occurrence.load_options(filename)
    assert opts is not None


@pytest.mark.skip(reason="not implemented")
def test_create_options_bundle():
    pass
