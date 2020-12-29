import os

import pytest
from penelope.co_occurrence import (
    ContextOpts,
    corpus_co_occurrence,
    folder_and_tag_to_filename,
    partitioned_corpus_co_occurrence,
    store_co_occurrences,
)
from penelope.co_occurrence.partitioned import ComputeResult
from penelope.corpus import SparvTokenizedCsvCorpus, TokensTransformOpts
from penelope.corpus.readers import ExtractTaggedTokensOpts, TextReaderOpts
from penelope.pipeline.interfaces import PipelinePayload
from penelope.utility import dataframe_to_tuples
from penelope.workflows import co_occurrence_workflow
from tests.test_data.corpus_fixtures import SIMPLE_CORPUS_ABCDEFG_3DOCS
from tests.utils import OUTPUT_FOLDER, TRANSTRÖMMER_ZIPPED_CSV_EXPORT_FILENAME, very_simple_corpus

jj = os.path.join


def test_co_occurrence_without_no_concept_and_threshold_succeeds():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    expected_result = [('c', 'b', 2), ('b', 'g', 1), ('b', 'f', 1), ('g', 'f', 1)]

    coo_df = corpus_co_occurrence(
        stream=corpus,
        payload=PipelinePayload(effective_document_index=corpus.document_index, token2id=corpus.token2id),
        context_opts=ContextOpts(concept={'b'}, ignore_concept=False, context_width=1),
        threshold_count=0,
    )
    assert expected_result == dataframe_to_tuples(coo_df, ['w1', 'w2', 'value'])


def test_co_occurrence_with_no_concept_succeeds():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)

    expected_result = {('d', 'a', 1), ('b', 'a', 1)}

    coo_df = corpus_co_occurrence(
        stream=corpus,
        payload=PipelinePayload(effective_document_index=corpus.document_index, token2id=corpus.token2id),
        context_opts=ContextOpts(concept={'g'}, ignore_concept=True, context_width=1),
        threshold_count=1,
    )
    assert expected_result == set(dataframe_to_tuples(coo_df, ['w1', 'w2', 'value']))


def test_co_occurrence_with_thresholdt_succeeds():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    expected_result = {('g', 'a', 2)}

    coo_df = corpus_co_occurrence(
        stream=corpus,
        payload=PipelinePayload(effective_document_index=corpus.document_index, token2id=corpus.token2id),
        context_opts=ContextOpts(concept={'g'}, ignore_concept=False, context_width=1),
        threshold_count=2,
    )
    assert expected_result == set(dataframe_to_tuples(coo_df, ['w1', 'w2', 'value']))


def test_co_occurrence_using_cli_succeeds(tmpdir):

    output_filename = folder_and_tag_to_filename(folder=tmpdir, tag='test_cli')
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

    co_occurrence_workflow(
        corpus_filename=TRANSTRÖMMER_ZIPPED_CSV_EXPORT_FILENAME,
        target_filename=output_filename,
        partition_keys=['year'],
        filename_field=["year:_:1"],
        context_opts=context_opts,
        extract_tokens_opts=extract_tokens_opts,
        tokens_transform_opts=tokens_transform_opts,
    )

    assert os.path.isfile(output_filename)


@pytest.mark.parametrize("concept, threshold_count, context_width", [({}, 0, 2), ({'fråga'}, 0, 2), ({'fråga'}, 2, 2)])
def test_partitioned_corpus_co_occurrence_succeeds(concept, threshold_count, context_width):

    corpus = SparvTokenizedCsvCorpus(
        './tests/test_data/riksdagens-protokoll.1920-2019.test.2files.zip',
        reader_opts=TextReaderOpts(
            filename_fields="year:_:1",
        ),
        extract_tokens_opts=ExtractTaggedTokensOpts(pos_includes='|NN|VB|', lemmatize=False),
    )

    compute_result: ComputeResult = partitioned_corpus_co_occurrence(
        stream=corpus,
        payload=PipelinePayload(effective_document_index=corpus.document_index, token2id=corpus.token2id),
        context_opts=ContextOpts(concept=concept, ignore_concept=False, context_width=context_width),
        global_threshold_count=threshold_count,
        partition_column='year',
    )

    assert compute_result is not None
    assert len(compute_result.co_occurrences) > 0


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
    compute_result: ComputeResult = partitioned_corpus_co_occurrence(
        stream=corpus,
        payload=PipelinePayload(effective_document_index=corpus.document_index, token2id=corpus.token2id),
        context_opts=ContextOpts(concept=concept, ignore_concept=False, context_width=n_context_width),
        global_threshold_count=None,
        partition_column='year',
    )

    assert compute_result is not None
    assert len(compute_result.co_occurrences) > 0


def test_co_occurrence_bug_with_options_that_raises_an_exception(tmpdir):

    target_filename = folder_and_tag_to_filename(folder=tmpdir, tag='test_co_occurrence_bug_with_option')
    options = {
        'corpus_filename': './tests/test_data/tranströmer_corpus_export.csv.zip',
        'target_filename': target_filename,
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
    co_occurrence_workflow(
        **options,
        context_opts=context_opts,
        extract_tokens_opts=extract_tokens_opts,
        tokens_transform_opts=tokens_transform_opts,
    )

    assert os.path.isfile(target_filename)


@pytest.mark.parametrize(
    'filename', ['partitioned_concept_co_occurrences_data.csv', 'partitioned_concept_co_occurrences_data.zip']
)
def test_store_when_co_occurrences_data_is_partitioned(filename):

    expected_filename = jj(OUTPUT_FOLDER, filename)
    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    compute_result: ComputeResult = partitioned_corpus_co_occurrence(
        stream=corpus,
        payload=PipelinePayload(effective_document_index=corpus.document_index, token2id=corpus.token2id),
        context_opts=ContextOpts(concept={'g'}, ignore_concept=False, context_width=2),
        global_threshold_count=1,
        partition_column='year',
    )

    store_co_occurrences(expected_filename, compute_result.co_occurrences)

    assert os.path.isfile(expected_filename)

    os.remove(expected_filename)
