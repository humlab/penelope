import os

import pytest
from penelope.co_occurrence import Bundle, ContextOpts
from penelope.co_occurrence.utility import compute_non_partitioned_corpus_co_occurrence
from penelope.corpus import ExtractTaggedTokensOpts, SparvTokenizedCsvCorpus, TextReaderOpts
from penelope.utility import dataframe_to_tuples

from ..fixtures import SIMPLE_CORPUS_ABCDEFG_3DOCS, very_simple_corpus

jj = os.path.join


def test_co_occurrence_without_no_concept_and_threshold_succeeds():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    expected_result = sorted([('b', 'g', 1), ('b', 'f', 1), ('c', 'b', 2), ('g', 'f', 1)])
    value: Bundle = compute_non_partitioned_corpus_co_occurrence(
        stream=corpus,
        token2id=corpus.token2id,
        document_index=corpus.document_index,
        context_opts=ContextOpts(concept={'b'}, ignore_concept=False, context_width=1),
        global_threshold_count=0,
    )
    assert expected_result == sorted(dataframe_to_tuples(value.decoded_co_occurrences[['w1', 'w2', 'value']]))


def test_co_occurrence_with_no_concept_succeeds():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)

    expected_result = sorted([('d', 'a', 1), ('b', 'a', 1)])

    value: Bundle = compute_non_partitioned_corpus_co_occurrence(
        stream=corpus,
        token2id=corpus.token2id,
        document_index=corpus.document_index,
        context_opts=ContextOpts(concept={'g'}, ignore_concept=True, ignore_padding=True, context_width=1),
        global_threshold_count=0,
    )
    assert expected_result == sorted(dataframe_to_tuples(value.decoded_co_occurrences[['w1', 'w2', 'value']]))


def test_co_occurrence_with_thresholdt_succeeds():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    expected_result = sorted([('g', 'a', 1), ('g', 'a', 1)])

    value: Bundle = compute_non_partitioned_corpus_co_occurrence(
        stream=corpus,
        token2id=corpus.token2id,
        document_index=corpus.document_index,
        context_opts=ContextOpts(concept={'g'}, ignore_concept=False, context_width=1),
        global_threshold_count=2,
    )
    assert expected_result == sorted(dataframe_to_tuples(value.decoded_co_occurrences[['w1', 'w2', 'value']]))


@pytest.mark.parametrize("concept, threshold_count, context_width", [({}, 0, 2), ({'fråga'}, 0, 2), ({'fråga'}, 2, 2)])
def test_compute_corpus_co_occurrence_succeeds(concept, threshold_count, context_width):

    corpus = SparvTokenizedCsvCorpus(
        './tests/test_data/riksdagens-protokoll.1920-2019.test.2files.zip',
        reader_opts=TextReaderOpts(
            filename_fields="year:_:1",
        ),
        extract_opts=ExtractTaggedTokensOpts(pos_includes='|NN|VB|', pos_paddings=None, lemmatize=False),
    )

    value: Bundle = compute_non_partitioned_corpus_co_occurrence(
        stream=corpus,
        document_index=corpus.document_index,
        token2id=corpus.token2id,
        context_opts=ContextOpts(concept=concept, ignore_concept=False, context_width=context_width),
        global_threshold_count=threshold_count,
    )

    assert value is not None
    assert len(value.co_occurrences) > 0


@pytest.mark.parametrize("concept, threshold_count, context_width", [({}, 0, 2), ({'fråga'}, 0, 2), ({'fråga'}, 2, 2)])
def test_partitioned_corpus_co_occurrence_succeeds2(concept, threshold_count, context_width):

    corpus = SparvTokenizedCsvCorpus(
        './tests/test_data/riksdagens-protokoll.1920-2019.test.2files.zip',
        reader_opts=TextReaderOpts(
            filename_fields="year:_:1",
        ),
        extract_opts=ExtractTaggedTokensOpts(pos_includes='|NN|VB|', pos_paddings=None, lemmatize=False),
    )

    value: Bundle = compute_non_partitioned_corpus_co_occurrence(
        stream=corpus,
        document_index=corpus.document_index,
        token2id=corpus.token2id,
        context_opts=ContextOpts(concept=concept, ignore_concept=False, context_width=context_width),
        global_threshold_count=threshold_count,
    )

    assert value is not None
    assert len(value.co_occurrences) > 0


@pytest.mark.long_running
def test_document_wise_co_occurrence():

    concept = set()
    n_context_width = 2
    corpus = SparvTokenizedCsvCorpus(
        # './tests/test_data/riksdagens-protokoll.1920-2019.9files.sparv4.csv.zip',
        './tests/test_data/riksdagens-protokoll.1920-2019.test.2files.zip',
        reader_opts=TextReaderOpts(
            filename_fields="year:_:1",
        ),
        extract_opts=ExtractTaggedTokensOpts(
            pos_includes='NN|PM|VB',
            pos_excludes='MAD|MID|PAD',
            pos_paddings="AB|DT|HA|HD|HP|HS|IE|IN|JJ|KN|PC|PL|PN|PP|PS|RG|RO|SN|UO",
            lemmatize=True,
        ),
    )
    bundle: Bundle = compute_non_partitioned_corpus_co_occurrence(
        stream=corpus,
        document_index=corpus.document_index,
        token2id=corpus.token2id,
        context_opts=ContextOpts(concept=concept, ignore_concept=False, context_width=n_context_width),
        global_threshold_count=None,
    )
    assert bundle is not None
    assert len(bundle.co_occurrences) > 0
