import numpy as np
import pandas as pd
import pytest
import scipy
from penelope.co_occurrence import Bundle, ContextOpts, CoOccurrenceHelper, to_filename
from penelope.co_occurrence.hal_or_glove.vectorizer_hal import HyperspaceAnalogueToLanguageVectorizer
from penelope.common.keyness import (
    KeynessMetric,
    compute_hal_cwr_score,
    compute_hal_score_by_co_occurrence_matrix,
    partitioned_significances,
)
from penelope.corpus import VectorizedCorpus
from tests.co_occurrence.utils import create_simple_bundle_by_pipeline

# pylint: disable=redefined-outer-name


@pytest.fixture(scope="module")
def bundle() -> Bundle:
    return create_bundle()


def create_bundle() -> Bundle:
    folder, tag = './tests/test_data/VENUS', 'VENUS'
    filename = to_filename(folder=folder, tag=tag)
    bundle: Bundle = Bundle.load(filename, compute_frame=False)
    return bundle


def test_burgess_litmus_test():
    terms = 'The Horse Raced Past The Barn Fell .'.lower().split()
    answer = {
        'barn': {'.': 4, 'barn': 0, 'fell': 5, 'horse': 0, 'past': 0, 'raced': 0, 'the': 0},
        'fell': {'.': 5, 'barn': 0, 'fell': 0, 'horse': 0, 'past': 0, 'raced': 0, 'the': 0},
        'horse': {'.': 0, 'barn': 2, 'fell': 1, 'horse': 0, 'past': 4, 'raced': 5, 'the': 3},
        'past': {'.': 2, 'barn': 4, 'fell': 3, 'horse': 0, 'past': 0, 'raced': 0, 'the': 5},
        'raced': {'.': 1, 'barn': 3, 'fell': 2, 'horse': 0, 'past': 5, 'raced': 0, 'the': 4},
        'the': {'.': 3, 'barn': 6, 'fell': 4, 'horse': 5, 'past': 3, 'raced': 4, 'the': 2},
    }
    df_answer = pd.DataFrame(answer).astype(np.uint32)[['the', 'horse', 'raced', 'past', 'barn', 'fell']].sort_index()
    # display(df_answer)
    vectorizer = HyperspaceAnalogueToLanguageVectorizer()
    vectorizer.fit([terms], size=5, distance_metric=0)
    df_imp = vectorizer.to_df().astype(np.uint32)[['the', 'horse', 'raced', 'past', 'barn', 'fell']].sort_index()
    assert df_imp.equals(df_answer), "Test failed"


def test_chen_lu_test():
    # Example in Chen, Lu:
    terms = 'The basic concept of the word association'.lower().split()
    vectorizer = HyperspaceAnalogueToLanguageVectorizer().fit([terms], size=5, distance_metric=0)
    df_imp = vectorizer.to_df().astype(np.uint32)[['the', 'basic', 'concept', 'of', 'word', 'association']].sort_index()
    df_answer = pd.DataFrame(
        {
            'the': [2, 5, 4, 3, 6, 4],
            'basic': [3, 0, 5, 4, 2, 1],
            'concept': [4, 0, 0, 5, 3, 2],
            'of': [5, 0, 0, 0, 4, 3],
            'word': [0, 0, 0, 0, 0, 5],
            'association': [0, 0, 0, 0, 0, 0],
        },
        index=['the', 'basic', 'concept', 'of', 'word', 'association'],
        dtype=np.uint32,
    ).sort_index()[['the', 'basic', 'concept', 'of', 'word', 'association']]

    assert df_imp.equals(df_answer), "Test failed"


def test_compute_hal_score_by_co_occurrence_matrix(bundle: Bundle):
    co_occurrences = bundle.co_occurrences
    co_occurrences['cwr'] = compute_hal_score_by_co_occurrence_matrix(
        bundle.co_occurrences, bundle.corpus.window_counts.document_term_window_counts
    )
    assert 'cwr' in co_occurrences.columns


def test_compute_hal_score_by_co_occurrence_matrix_burgess_litmus():
    data = [('document_01.txt', 'The Horse Raced Past The Barn Fell .'.lower().split())]
    context_opts: ContextOpts = ContextOpts(
        context_width=2,
        concept=set(),
    )
    bundle: Bundle = create_simple_bundle_by_pipeline(data, context_opts)
    co_occurrences = bundle.co_occurrences
    co_occurrences['cwr'] = compute_hal_score_by_co_occurrence_matrix(
        bundle.co_occurrences, bundle.corpus.window_counts.document_term_window_counts
    )
    assert 'cwr' in co_occurrences.columns


def test_HAL_cwr_corpus(bundle: Bundle):
    corpus: VectorizedCorpus = bundle.corpus
    nw_x = corpus.window_counts.document_term_window_counts  # .todense().astype(np.float)
    nw_xy = corpus.data  # .copy().astype(np.float)
    nw_cwr: scipy.sparse.spmatrix = compute_hal_cwr_score(nw_xy, nw_x, corpus.vocabs_mapping)

    assert nw_cwr is not None
    assert nw_cwr.sum() > 0

    hal_cwr_corpus: VectorizedCorpus = corpus.to_HAL_cwr_keyness()
    assert hal_cwr_corpus.data.sum() == nw_cwr.sum()


# def test_HAL_cwr_corpus_bug():

#     folder, tag = '/home/roger/source/welfare-state-analytics/welfare_state_analytics/data/TEST-information/', 'TEST-information'
#     filename = to_filename(folder=folder, tag=tag)
#     bundle: Bundle = Bundle.load(filename, compute_frame=False)

#     corpus: VectorizedCorpus = bundle.corpus
#     nw_x = corpus.window_counts.document_term_window_counts  # .todense().astype(np.float)
#     nw_xy = corpus.data  # .copy().astype(np.float)
#     nw_cwr: scipy.sparse.spmatrix = compute_hal_cwr_score(nw_xy, nw_x, corpus.vocabs_mapping)

#     assert nw_cwr is not None
#     assert nw_cwr.sum() > 0

#     hal_cwr_corpus: VectorizedCorpus = corpus.to_HAL_cwr_keyness()
#     assert hal_cwr_corpus.data.sum() == nw_cwr.sum()


def test_HAL_cwr_corpus_burgess_litmus():
    data = [('document_01.txt', 'The Horse Raced Past The Barn Fell .'.lower().split())]
    context_opts: ContextOpts = ContextOpts(
        context_width=2,
        concept=set(),
        ignore_padding=False,
    )
    bundle: Bundle = create_simple_bundle_by_pipeline(data, context_opts)

    hal_cwr_corpus: VectorizedCorpus = bundle.corpus.to_HAL_cwr_keyness()

    assert hal_cwr_corpus is not None


def create_helper(bundle: Bundle, period_specifier: str, pivot_key: str):
    corpus: VectorizedCorpus = bundle.corpus.group_by_time_period_optimized(
        time_period_specifier=period_specifier,
        target_column_name=pivot_key,
    )

    co_occurrences = corpus.to_co_occurrences(token2id=bundle.token2id, partition_key=pivot_key)

    helper: CoOccurrenceHelper = CoOccurrenceHelper(
        corpus=bundle.corpus,
        source_token2id=bundle.token2id,
        pivot_keys=[pivot_key],
        co_occurrences=co_occurrences,
    )
    return helper


@pytest.mark.parametrize(
    'keyness', [KeynessMetric.PPMI, KeynessMetric.DICE, KeynessMetric.LLR, KeynessMetric.LLR_Dunning]
)
def test_compute_significance(bundle: Bundle, keyness: KeynessMetric):  # pylint: disable=unused-argument
    document_pivot_key = 'year'
    pivot_key = 'time_period'

    helper: CoOccurrenceHelper = create_helper(bundle, document_pivot_key, pivot_key)
    co_occurrences = helper.value
    vocabulary_size = max(bundle.token2id.values()) + 1

    weighed_co_occurrences = partitioned_significances(
        co_occurrences,
        pivot_key=pivot_key,
        keyness_metric=keyness,
        document_index=helper.corpus.document_index,
        vocabulary_size=vocabulary_size,
        normalize=False,
    )

    assert weighed_co_occurrences is not None
