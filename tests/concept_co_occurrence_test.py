import os
from tests.test_data.corpus_fixtures import SIMPLE_CORPUS_ABCDEFG_3DOCS

import pytest

from penelope.co_occurrence import (
    partitioned_corpus_concept_co_occurrence,
)
from penelope.co_occurrence.concept_co_occurrence import corpus_concept_co_occurrence
from penelope.corpus import SparvTokenizedCsvCorpus
from penelope.scripts.concept_co_occurrence import cli_concept_co_occurrence
from penelope.utility import dataframe_to_tuples

from .utils import TRANSTRÖMMER_ZIPPED_CSV_EXPORT_FILENAME, very_simple_corpus

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

    cli_concept_co_occurrence(**options)

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

    cli_concept_co_occurrence(**options)

    assert os.path.isfile(output_filename)


def test_load_co_occurrences_data():
    pass

def test_vectorize_co_occurrences_data():
    pass
