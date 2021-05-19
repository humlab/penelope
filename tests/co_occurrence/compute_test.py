import os

import numpy as np
import pytest
from penelope.co_occurrence import Bundle, ContextOpts, CoOccurrenceComputeResult
from penelope.co_occurrence.partition_by_document import (
    co_occurrences_to_vectorized_corpus,
    compute_corpus_co_occurrence,
)
from penelope.co_occurrence.persistence import to_filename
from penelope.corpus import CorpusVectorizer, ExtractTaggedTokensOpts, SparvTokenizedCsvCorpus, TextReaderOpts, Token2Id
from penelope.utility import dataframe_to_tuples
from tests.fixtures import SIMPLE_CORPUS_ABCDE_5DOCS, SIMPLE_CORPUS_ABCDEFG_3DOCS, very_simple_corpus
from tests.utils import OUTPUT_FOLDER

jj = os.path.join


def test_co_occurrence_matrix_of_corpus_returns_correct_result():

    expected_token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
    expected_matrix = np.matrix([[0, 6, 4, 3, 3], [0, 0, 2, 1, 4], [0, 0, 0, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)

    v_corpus = CorpusVectorizer().fit_transform(corpus, already_tokenized=True, vocabulary=corpus.token2id)

    term_term_matrix = v_corpus.co_occurrence_matrix()

    assert (term_term_matrix.todense() == expected_matrix).all()
    assert expected_token2id == v_corpus.token2id


# FIXME Add document_id to results & asserts
def test_co_occurrence_without_no_concept_and_threshold_succeeds():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    expected_result = sorted([('b', 'g', 1), ('b', 'f', 1), ('c', 'b', 2), ('g', 'f', 1)])
    value: CoOccurrenceComputeResult = compute_corpus_co_occurrence(
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

    value: CoOccurrenceComputeResult = compute_corpus_co_occurrence(
        stream=corpus,
        token2id=corpus.token2id,
        document_index=corpus.document_index,
        context_opts=ContextOpts(concept={'g'}, ignore_concept=True, context_width=1),
        global_threshold_count=0,
    )
    assert expected_result == sorted(dataframe_to_tuples(value.decoded_co_occurrences[['w1', 'w2', 'value']]))


def test_co_occurrence_with_thresholdt_succeeds():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    expected_result = sorted([('g', 'a', 1), ('g', 'a', 1)])

    value: CoOccurrenceComputeResult = compute_corpus_co_occurrence(
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

    value: CoOccurrenceComputeResult = compute_corpus_co_occurrence(
        stream=corpus,
        document_index=corpus.document_index,
        token2id=corpus.token2id,
        context_opts=ContextOpts(concept=concept, ignore_concept=False, context_width=context_width),
        global_threshold_count=threshold_count,
    )

    assert value is not None
    assert len(value.co_occurrences) > 0


def test_store_complete_bundle_compute_corpus_co_occurrence():

    tag: str = "JUPYTER"
    folder: str = jj(OUTPUT_FOLDER, tag)
    filename: str = to_filename(folder=folder, tag=tag)

    os.makedirs(folder, exist_ok=True)

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    token2id: Token2Id = Token2Id(corpus.token2id)

    value: CoOccurrenceComputeResult = compute_corpus_co_occurrence(
        stream=corpus,
        document_index=corpus.document_index,
        token2id=token2id,
        context_opts=ContextOpts(concept={'g'}, ignore_concept=False, context_width=2),
        global_threshold_count=1,
    )

    corpus = co_occurrences_to_vectorized_corpus(
        co_occurrences=value.co_occurrences,
        document_index=value.document_index,
        token2id=token2id,
    )

    bundle: Bundle = Bundle(
        folder=folder,
        tag=tag,
        co_occurrences=value.co_occurrences,
        document_index=value.document_index,
        token_window_counts=value.token_window_counts,
        token2id=value.token2id,
        compute_options={},
        corpus=corpus,
    )

    bundle.store()

    assert os.path.isfile(filename)

    # os.remove(expected_filename)


@pytest.mark.parametrize("concept, threshold_count, context_width", [({}, 0, 2), ({'fråga'}, 0, 2), ({'fråga'}, 2, 2)])
def test_partitioned_corpus_co_occurrence_succeeds2(concept, threshold_count, context_width):

    corpus = SparvTokenizedCsvCorpus(
        './tests/test_data/riksdagens-protokoll.1920-2019.test.2files.zip',
        reader_opts=TextReaderOpts(
            filename_fields="year:_:1",
        ),
        extract_opts=ExtractTaggedTokensOpts(pos_includes='|NN|VB|', pos_paddings=None, lemmatize=False),
    )

    value: CoOccurrenceComputeResult = compute_corpus_co_occurrence(
        stream=corpus,
        document_index=corpus.document_index,
        token2id=corpus.token2id,
        context_opts=ContextOpts(concept=concept, ignore_concept=False, context_width=context_width),
        global_threshold_count=threshold_count,
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
    )

    assert value is not None

    # dtm_corpus = co_occurrences_to_vectorized_corpus(
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

    # bundle.store()

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
        extract_opts=ExtractTaggedTokensOpts(
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
    )

    assert value is not None
    assert len(value.co_occurrences) > 0
