import os

import pytest
from penelope.co_occurrence import (
    ContextOpts,
    corpus_co_occurrence,
    load_co_occurrences,
    partitioned_corpus_co_occurrence,
    store_co_occurrences,
    to_vectorized_corpus,
)
from penelope.corpus import SparvTokenizedCsvCorpus, TokensTransformOpts
from penelope.corpus.readers import ExtractTaggedTokensOpts, TextReaderOpts
from penelope.utility import dataframe_to_tuples, pretty_print_matrix
from penelope.workflows import concept_co_occurrence_workflow
from tests.test_data.corpus_fixtures import SIMPLE_CORPUS_ABCDEFG_3DOCS

from .utils import OUTPUT_FOLDER, TEST_DATA_FOLDER, TRANSTRÖMMER_ZIPPED_CSV_EXPORT_FILENAME, very_simple_corpus

jj = os.path.join


def test_concept_co_occurrence_without_no_concept_and_threshold_succeeds():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    expected_result = [('c', 'b', 2), ('b', 'g', 1), ('b', 'f', 1), ('g', 'f', 1)]

    coo_df = corpus_co_occurrence(
        stream=corpus,
        document_index=corpus.documents,
        token2id=corpus.token2id,
        context_opts=ContextOpts(concept={'b'}, ignore_concept=False, context_width=1),
        threshold_count=0,
    )
    assert expected_result == dataframe_to_tuples(coo_df, ['w1', 'w2', 'value'])


def test_concept_co_occurrence_with_no_concept_succeeds():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)

    expected_result = {('d', 'a', 1), ('b', 'a', 1)}

    coo_df = corpus_co_occurrence(
        stream=corpus,
        document_index=corpus.documents,
        token2id=corpus.token2id,
        context_opts=ContextOpts(concept={'g'}, ignore_concept=True, context_width=1),
        threshold_count=1,
    )
    assert expected_result == set(dataframe_to_tuples(coo_df, ['w1', 'w2', 'value']))


def test_concept_co_occurrence_with_thresholdt_succeeds():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    expected_result = {('g', 'a', 2)}

    coo_df = corpus_co_occurrence(
        stream=corpus,
        document_index=corpus.documents,
        token2id=corpus.token2id,
        context_opts=ContextOpts(concept={'g'}, ignore_concept=False, context_width=1),
        threshold_count=2,
    )
    assert expected_result == set(dataframe_to_tuples(coo_df, ['w1', 'w2', 'value']))


def test_co_occurrence_using_cli_succeeds(tmpdir):

    output_filename = jj(tmpdir, 'test_co_occurrence_using_cli_succeeds.csv')
    context_opts = ContextOpts(
        concept={'jag'},
        context_width=2,
    )
    extract_tokens_opts = ExtractTaggedTokensOpts(pos_includes=None, pos_excludes='|MAD|MID|PAD|', lemmatize=True)
    tokens_transform_opts = TokensTransformOpts(
        remove_stopwords=None,
        keep_symbols=True,
        keep_numerals=True,
        only_alphabetic=False,
        only_any_alphanumeric=False,
    )

    concept_co_occurrence_workflow(
        input_filename=TRANSTRÖMMER_ZIPPED_CSV_EXPORT_FILENAME,
        output_filename=output_filename,
        partition_keys=['year'],
        filename_field=["year:_:1"],
        context_opts=context_opts,
        extract_tokens_opts=extract_tokens_opts,
        tokens_transform_opts=tokens_transform_opts,
    )

    assert os.path.isfile(output_filename)


@pytest.mark.parametrize("concept, threshold_count, context_width", [('fråga', 0, 2), ('fråga', 2, 2)])
def test_partitioned_corpus_co_occurrence_succeeds(concept, threshold_count, context_width):

    corpus = SparvTokenizedCsvCorpus(
        './tests/test_data/riksdagens-protokoll.1920-2019.test.2files.zip',
        reader_opts=TextReaderOpts(
            filename_fields="year:_:1",
        ),
        extract_tokens_opts=ExtractTaggedTokensOpts(pos_includes='|NN|VB|', lemmatize=False),
    )

    coo_df = partitioned_corpus_co_occurrence(
        stream=corpus,
        document_index=corpus.documents,
        token2id=corpus.token2id,
        context_opts=ContextOpts(concept={concept}, ignore_concept=False, context_width=context_width),
        global_threshold_count=threshold_count,
        partition_column='year',
    )

    assert coo_df is not None
    assert len(coo_df) > 0


@pytest.mark.skip("long running, used for bug fixes")
def test_co_occurrence_of_windowed_corpus_returns_correct_result4():

    concept = {'jag'}
    n_context_width = 2
    corpus = SparvTokenizedCsvCorpus(
        './tests/test_data/riksdagens-protokoll.1920-2019.test.zip',
        reader_opts=TextReaderOpts(
            filename_fields="year:_:1",
        ),
        extract_tokens_opts=ExtractTaggedTokensOpts(pos_includes='|NN|VB|', lemmatize=False),
    )
    coo_df = partitioned_corpus_co_occurrence(
        stream=corpus,
        document_index=corpus.documents,
        token2id=corpus.token2id,
        context_opts=ContextOpts(concept=concept, ignore_concept=False, context_width=n_context_width),
        global_threshold_count=None,
        partition_column='year',
    )

    assert coo_df is not None
    assert len(coo_df) > 0


def test_co_occurrence_bug_with_options_that_raises_an_exception(tmpdir):

    output_filename = jj(tmpdir, 'test_co_occurrence_bug_with_options_that_raises_an_exception.csv')
    options = {
        'input_filename': './tests/test_data/tranströmer_corpus_export.csv.zip',
        'output_filename': output_filename,
        'partition_keys': ('year',),
        'filename_field': ('year:_:1',),
    }
    context_opts = ContextOpts(concept=('jag',), context_width=2)
    extract_tokens_opts = ExtractTaggedTokensOpts(pos_includes=None, pos_excludes='|MAD|MID|PAD|', lemmatize=False)
    tokens_transform_opts = TokensTransformOpts(
        to_lower=True,
        min_len=1,
        remove_stopwords=None,
        keep_symbols=True,
        keep_numerals=True,
        only_alphabetic=False,
        only_any_alphanumeric=False,
    )
    concept_co_occurrence_workflow(
        **options,
        context_opts=context_opts,
        extract_tokens_opts=extract_tokens_opts,
        tokens_transform_opts=tokens_transform_opts,
    )

    assert os.path.isfile(output_filename)


@pytest.mark.parametrize('filename', ['concept_co_occurrences_data.csv', 'concept_co_occurrences_data.zip'])
def test_store_when_co_occurrences_data_is_not_partitioned(filename):

    expected_filename = jj(OUTPUT_FOLDER, filename)
    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    coo_df = corpus_co_occurrence(
        stream=corpus,
        document_index=corpus.documents,
        token2id=corpus.token2id,
        context_opts=ContextOpts(concept={'g'}, ignore_concept=False, context_width=2),
        threshold_count=1,
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
    df = partitioned_corpus_co_occurrence(
        stream=corpus,
        document_index=corpus.documents,
        token2id=corpus.token2id,
        context_opts=ContextOpts(concept={'g'}, ignore_concept=False, context_width=2),
        global_threshold_count=1,
        partition_column='year',
    )

    store_co_occurrences(expected_filename, df)

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
