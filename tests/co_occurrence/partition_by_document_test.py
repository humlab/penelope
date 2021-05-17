import os

import pytest
from penelope.co_occurrence import Bundle, ContextOpts, CoOccurrenceComputeResult, store_bundle
from penelope.co_occurrence.partition_by_document import (
    co_occurrence_dataframe_to_vectorized_corpus,
    compute_corpus_co_occurrence,
)
from penelope.corpus import ExtractTaggedTokensOpts, SparvTokenizedCsvCorpus, TextReaderOpts
from tests.test_data.corpus_fixtures import SIMPLE_CORPUS_ABCDEFG_3DOCS
from tests.utils import OUTPUT_FOLDER, very_simple_corpus

jj = os.path.join


@pytest.mark.parametrize("concept, threshold_count, context_width", [({}, 0, 2), ({'fråga'}, 0, 2), ({'fråga'}, 2, 2)])
def test_partitioned_corpus_co_occurrence_succeeds(concept, threshold_count, context_width):

    corpus = SparvTokenizedCsvCorpus(
        './tests/test_data/riksdagens-protokoll.1920-2019.test.2files.zip',
        reader_opts=TextReaderOpts(
            filename_fields="year:_:1",
        ),
        extract_tokens_opts=ExtractTaggedTokensOpts(pos_includes='|NN|VB|', pos_paddings=None, lemmatize=False),
    )

    value: CoOccurrenceComputeResult = compute_corpus_co_occurrence(
        stream=corpus,
        document_index=corpus.document_index,
        token2id=corpus.token2id,
        context_opts=ContextOpts(concept=concept, ignore_concept=False, context_width=context_width),
        global_threshold_count=threshold_count,
        ignore_pad=None,
    )

    assert value is not None
    assert len(value.co_occurrences) > 0


@pytest.mark.parametrize(
    'filename', ['partitioned_concept_co_occurrences_data.csv', 'partitioned_concept_co_occurrences_data.zip']
)
def test_store_when_co_occurrences_data_is_partitioned(filename):

    expected_filename = jj(OUTPUT_FOLDER, filename)
    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    value: CoOccurrenceComputeResult = compute_corpus_co_occurrence(
        stream=corpus,
        document_index=corpus.document_index,
        token2id=corpus.token2id,
        context_opts=ContextOpts(concept={'g'}, ignore_concept=False, context_width=2),
        global_threshold_count=1,
        ignore_pad=None,
    )

    dtm_corpus = co_occurrence_dataframe_to_vectorized_corpus(
        co_occurrences=value.co_occurrences,
        document_index=value.document_index,
        token2id=corpus.token2id,
    )

    bundle: Bundle = Bundle(
        co_occurrences=value.co_occurrences,
        document_index=value.document_index,
        co_occurrences_filename=expected_filename,
        compute_options={},
        corpus=dtm_corpus,
        corpus_folder='./tests/output',
        corpus_tag='partitioned_concept_co_occurrences_data',
    )

    store_bundle(expected_filename, bundle)

    assert os.path.isfile(expected_filename)

    # os.remove(expected_filename)


@pytest.mark.parametrize("concept, threshold_count, context_width", [({}, 0, 2), ({'fråga'}, 0, 2), ({'fråga'}, 2, 2)])
def test_partitioned_corpus_co_occurrence_succeeds2(concept, threshold_count, context_width):

    corpus = SparvTokenizedCsvCorpus(
        './tests/test_data/riksdagens-protokoll.1920-2019.test.2files.zip',
        reader_opts=TextReaderOpts(
            filename_fields="year:_:1",
        ),
        extract_tokens_opts=ExtractTaggedTokensOpts(pos_includes='|NN|VB|', pos_paddings=None, lemmatize=False),
    )

    value: CoOccurrenceComputeResult = compute_corpus_co_occurrence(
        stream=corpus,
        document_index=corpus.document_index,
        token2id=corpus.token2id,
        context_opts=ContextOpts(concept=concept, ignore_concept=False, context_width=context_width),
        global_threshold_count=threshold_count,
        ignore_pad=None,
    )

    assert value is not None
    assert len(value.co_occurrences) > 0


@pytest.mark.parametrize(
    'filename', ['partitioned_concept_co_occurrences_data.csv', 'partitioned_concept_co_occurrences_data.zip']
)
def test_create_document_co_occurrences(filename):  # pylint: disable=unused-argument

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    value: CoOccurrenceComputeResult = compute_corpus_co_occurrence(
        stream=corpus,
        token2id=corpus.token2id,
        document_index=corpus.document_index,
        context_opts=ContextOpts(concept={'g'}, ignore_concept=False, context_width=2),
        global_threshold_count=1,
        ignore_pad=None,
    )

    assert value is not None

    # dtm_corpus = co_occurrence_dataframe_to_vectorized_corpus(
    #     co_occurrences=value.co_occurrences,
    #     token2id=corpus.token2id,
    #     document_index=value.document_index,
    # )

    # bundle: Bundle = Bundle(
    #     co_occurrences=value.co_occurrences,
    #     document_index=value.document_index,
    #     co_occurrences_filename=expected_filename,
    #     compute_options={},
    #     corpus=dtm_corpus,
    #     corpus_folder='./tests/output',
    #     corpus_tag='partitioned_concept_co_occurrences_data',
    # )

    # store_bundle(expected_filename, bundle)

    # assert os.path.isfile(expected_filename)

    # os.remove(expected_filename)


def test_document_wise_co_occurrence():

    concept = {}
    n_context_width = 2
    corpus = SparvTokenizedCsvCorpus(
        # './tests/test_data/riksdagens-protokoll.1920-2019.9files.sparv4.csv.zip',
        './tests/test_data/riksdagens-protokoll.1920-2019.test.zip',
        reader_opts=TextReaderOpts(
            filename_fields="year:_:1",
        ),
        extract_tokens_opts=ExtractTaggedTokensOpts(
            pos_includes='NN|PM|VB',
            pos_excludes='MAD|MID|PAD',
            pos_paddings="AB|DT|HA|HD|HP|HS|IE|IN|JJ|KN|PC|PL|PN|PP|PS|RG|RO|SN|UO",
            lemmatize=True,
        ),
    )
    value: CoOccurrenceComputeResult = compute_corpus_co_occurrence(
        stream=corpus,
        document_index=corpus.document_index,
        token2id=corpus.token2id,
        context_opts=ContextOpts(concept=concept, ignore_concept=False, context_width=n_context_width),
        global_threshold_count=None,
        ignore_pad=None,
    )

    assert value is not None
    assert len(value.co_occurrences) > 0
