import os

import pytest
from penelope.co_occurrence import partitioned_corpus_concept_co_occurrence
from penelope.co_occurrence.concept_co_occurrence import (
    corpus_concept_co_occurrence,
    load_co_occurrences,
    store_co_occurrences,
    to_vectorized_corpus,
)
from penelope.corpus import SparvTokenizedCsvCorpus
from penelope.utility import dataframe_to_tuples, pretty_print_matrix
from penelope.workflows import execute_workflow_concept_co_occurrence
from tests.test_data.corpus_fixtures import SIMPLE_CORPUS_ABCDEFG_3DOCS

from .utils import OUTPUT_FOLDER, TEST_DATA_FOLDER, TRANSTRÖMMER_ZIPPED_CSV_EXPORT_FILENAME, very_simple_corpus

jj = os.path.join


def test_concept_co_occurrence_without_no_concept_and_threshold_succeeds():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    expected_result = [('c', 'b', 2), ('b', 'g', 1), ('b', 'f', 1), ('g', 'f', 1)]

    coo_df = corpus_concept_co_occurrence(
        corpus, concepts={'b'}, no_concept=False, n_count_threshold=0, n_context_width=1
    )
    assert expected_result == dataframe_to_tuples(coo_df, ['w1', 'w2', 'value'])


def test_concept_co_occurrence_with_no_concept_succeeds():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)

    expected_result = {('d', 'a', 1), ('b', 'a', 1)}

    coo_df = corpus_concept_co_occurrence(
        corpus, concepts={'g'}, no_concept=True, n_count_threshold=1, n_context_width=1
    )
    assert expected_result == set(dataframe_to_tuples(coo_df, ['w1', 'w2', 'value']))


def test_concept_co_occurrence_with_thresholdt_succeeds():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    expected_result = {('g', 'a', 2)}

    coo_df = corpus_concept_co_occurrence(
        corpus, concepts={'g'}, no_concept=False, n_count_threshold=2, n_context_width=1
    )
    assert expected_result == set(dataframe_to_tuples(coo_df, ['w1', 'w2', 'value']))


def test_co_occurrence_using_cli_succeeds(tmpdir):

    output_filename = jj(tmpdir, 'test_co_occurrence_using_cli_succeeds.csv')
    options = dict(
        input_filename=TRANSTRÖMMER_ZIPPED_CSV_EXPORT_FILENAME,
        output_filename=output_filename,
        concept={'jag'},
        context_width=2,
        partition_keys=['year'],
        pos_includes=None,
        pos_excludes='|MAD|MID|PAD|',
        lemmatize=True,
        to_lowercase=True,
        remove_stopwords=None,
        min_word_length=1,
        keep_symbols=True,
        keep_numerals=True,
        only_alphabetic=False,
        only_any_alphanumeric=False,
        filename_field=["year:_:1"],
    )

    execute_workflow_concept_co_occurrence(**options)

    assert os.path.isfile(output_filename)


@pytest.mark.parametrize("concept, n_count_threshold, n_context_width", [('fråga', 0, 2), ('fråga', 2, 2)])
def test_partitioned_corpus_concept_co_occurrence_succeeds(concept, n_count_threshold, n_context_width):

    corpus = SparvTokenizedCsvCorpus(
        './tests/test_data/riksdagens-protokoll.1920-2019.test.2files.zip',
        tokenizer_opts=dict(
            filename_fields="year:_:1",
        ),
        pos_includes='|NN|VB|',
        lemmatize=False,
    )

    coo_df = partitioned_corpus_concept_co_occurrence(
        corpus,
        concepts={concept},
        no_concept=False,
        n_count_threshold=n_count_threshold,
        n_context_width=n_context_width,
        partition_keys='year',
    )

    assert coo_df is not None
    assert len(coo_df) > 0


@pytest.mark.skip("long running, used for bug fixes")
def test_co_occurrence_of_windowed_corpus_returns_correct_result4():

    concept = {'jag'}
    n_context_width = 2
    corpus = SparvTokenizedCsvCorpus(
        './tests/test_data/riksdagens-protokoll.1920-2019.test.zip',
        tokenizer_opts=dict(
            filename_fields="year:_:1",
        ),
        pos_includes='|NN|VB|',
        lemmatize=False,
    )
    coo_df = partitioned_corpus_concept_co_occurrence(
        corpus,
        concepts=concept,
        no_concept=False,
        n_count_threshold=None,
        n_context_width=n_context_width,
        partition_keys='year',
    )

    assert coo_df is not None
    assert len(coo_df) > 0


def test_co_occurrence_bug_with_options_that_raises_an_exception(tmpdir):

    output_filename = jj(tmpdir, 'test_co_occurrence_bug_with_options_that_raises_an_exception.csv')
    options = {
        'input_filename': './tests/test_data/tranströmer_corpus_export.csv.zip',
        'output_filename': output_filename,
        'concept': ('jag',),
        'context_width': 2,
        'partition_keys': ('year',),
        'pos_includes': None,
        'pos_excludes': '|MAD|MID|PAD|',
        'lemmatize': True,
        'to_lowercase': True,
        'remove_stopwords': None,
        'min_word_length': 1,
        'keep_symbols': True,
        'keep_numerals': True,
        'only_alphabetic': False,
        'only_any_alphanumeric': False,
        'filename_field': ('year:_:1',),
    }

    execute_workflow_concept_co_occurrence(**options)

    assert os.path.isfile(output_filename)


@pytest.mark.parametrize('filename', ['concept_co_occurrences_data.csv', 'concept_co_occurrences_data.zip'])
def test_store_when_co_occurrences_data_is_not_partitioned(filename):

    expected_filename = jj(OUTPUT_FOLDER, filename)
    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    coo_df = corpus_concept_co_occurrence(
        corpus, concepts={'g'}, no_concept=False, n_count_threshold=1, n_context_width=2
    )

    store_co_occurrences(expected_filename, coo_df)

    assert os.path.isfile(expected_filename)

    os.remove(expected_filename)


@pytest.mark.parametrize('filename', ['concept_co_occurrences_data.csv', 'concept_co_occurrences_data.zip'])
def test_load_when_co_occurrences_data_is_not_partitioned(filename):

    filename = jj(TEST_DATA_FOLDER, filename)

    df = load_co_occurrences(filename)

    assert df is not None
    assert 13 == len(df)
    assert 18 == df.value.sum()
    assert 3 == int(df[(df.w1 == 'g') & (df.w2 == 'a')]['value'])
    assert (['w1', 'w2', 'value', 'value_n_d', 'value_n_t'] == df.columns).all()


@pytest.mark.parametrize(
    'filename', ['partitioned_concept_co_occurrences_data.csv', 'partitioned_concept_co_occurrences_data.zip']
)
def test_store_when_co_occurrences_data_is_partitioned(filename):

    expected_filename = jj(OUTPUT_FOLDER, filename)
    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    coo_df = partitioned_corpus_concept_co_occurrence(
        corpus, concepts={'g'}, no_concept=False, n_count_threshold=1, n_context_width=2, partition_keys='year'
    )

    store_co_occurrences(expected_filename, coo_df)

    assert os.path.isfile(expected_filename)

    os.remove(expected_filename)


def test_vectorize_co_occurrences_data():

    value_column = 'value_n_t'
    filename = jj(TEST_DATA_FOLDER, 'partitioned_concept_co_occurrences_data.zip')

    co_occurrences = load_co_occurrences(filename)

    v_corpus = to_vectorized_corpus(co_occurrences, value_column)

    pretty_print_matrix(
        v_corpus.data.todense(),
        row_labels=[str(i) for i in v_corpus.documents.year],
        column_labels=v_corpus.vocabulary,
        dtype=float,
        float_fmt="{0:.04f}",
    )

    assert v_corpus is not None
