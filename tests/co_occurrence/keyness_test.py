# type: ignore
# pylint: disable=unused-import

from pprint import pprint as pp
from typing import Sequence

import numpy as np
import pytest
import scipy
from penelope.co_occurrence import Bundle, ContextOpts
from penelope.co_occurrence.keyness import ComputeKeynessOpts, compute_corpus_keyness, significance_ratio
from penelope.common.keyness import KeynessMetric, KeynessMetricSource, metrics
from penelope.corpus import VectorizedCorpus
from tests.utils import incline_code

from .utils import create_keyness_opts, create_keyness_test_bundle, create_simple_bundle_by_pipeline

SIMPLE_CORPUS_ABCDE_3DOCS = [
    ('tran_2019_01_test.txt', ['a', 'b', 'c', 'c', 'd', 'c', 'e']),
    ('tran_2019_02_test.txt', ['a', 'a', 'c', 'e', 'c', 'd', 'd']),
    ('tran_2019_03_test.txt', ['d', 'e', 'e', 'b']),
]

# pylint: disable=protected-access


def test_keyness_transform_with_simple_corpus():

    bundle: Bundle = create_keyness_test_bundle(data=SIMPLE_CORPUS_ABCDE_3DOCS)
    opts: ComputeKeynessOpts = create_keyness_opts()

    corpus: VectorizedCorpus = bundle.keyness_transform(opts=opts)

    assert corpus is not None


def test_step_by_step_tfidf_keyness_transform():

    bundle: Bundle = create_keyness_test_bundle(data=SIMPLE_CORPUS_ABCDE_3DOCS, processes=None, ignore_padding=False)
    opts: ComputeKeynessOpts = create_keyness_opts()

    corpus: VectorizedCorpus = bundle.corpus
    concept_corpus: VectorizedCorpus = bundle.concept_corpus

    """ STEP: Reduce corpus size if TF threshold is specified
        @filename: keyness.py, compute_weighed_corpus_keyness:75"""

    assert corpus.term_frequency.tolist() == [3, 2, 2, 1, 3, 4, 8, 3, 5, 6, 3, 1, 3]
    zero_out_indices: Sequence[int] = corpus.zero_out_by_tf_threshold(3)
    assert zero_out_indices.tolist() == [1, 2, 3, 11]
    assert corpus.term_frequency.tolist() == [3, 0, 0, 0, 3, 4, 8, 3, 5, 6, 3, 0, 3]

    assert concept_corpus.term_frequency.tolist() == [0, 0, 0, 0, 0, 0, 8, 1, 5, 2, 3, 0, 0]
    concept_corpus.zero_out_by_indices(zero_out_indices)
    assert concept_corpus.term_frequency.tolist() == [0, 0, 0, 0, 0, 0, 8, 1, 5, 2, 3, 0, 0]

    """ STEP: Compute corpus keyness for both corpora
        @filename: penelope/co_occurrence/keyness.py:23, compute_corpus_keyness
        Compute keyness (TF-IDF in this case - must be done before grouping)
    """
    with incline_code(source=compute_corpus_keyness):
        assert (
            (
                corpus.data.todense()
                == np.matrix(
                    [
                        [1, 0, 0, 0, 3, 1, 5, 1, 1, 2, 0, 0, 0],
                        [2, 0, 0, 0, 0, 3, 3, 0, 1, 4, 2, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 1, 0, 3],
                    ],
                    dtype=np.int32,
                )
            )
            .all()
            .all()
        )

        corpus = corpus.tf_idf()
        corpus = corpus.group_by_time_period_optimized(
            time_period_specifier=opts.period_pivot,
            target_column_name=opts.pivot_column_name,
            fill_gaps=opts.fill_gaps,
        )

        concept_corpus = concept_corpus.tf_idf()
        concept_corpus = concept_corpus.group_by_time_period_optimized(
            time_period_specifier=opts.period_pivot,
            target_column_name=opts.pivot_column_name,
            fill_gaps=opts.fill_gaps,
        )

        M: scipy.sparse.spmatrix = significance_ratio(concept_corpus.data, corpus.data)

    assert M is not None


def test_significant_ratio():
    A = scipy.sparse.csr_matrix(np.array([[1, 2, 4], [2, 4, 5], [25, 15, 20]]))
    B = A.copy()

    R = metrics.significance_ratio(A, B)

    assert (R == 1.0).todense().all().all()

    B = scipy.sparse.csr_matrix(np.array([[1, 2, 2], [4, 1, 25], [5, 5, 0]]))
    R = metrics.significance_ratio(A, B)
    assert (
        (R.todense() == scipy.sparse.csr_matrix(np.array([[1.0, 1.0, 2.0], [0.5, 4.0, 0.2], [5.0, 3.0, 0.0]])))
        .all()
        .all()
    )


@pytest.mark.parametrize(
    "tag,keyness_source,keyness",
    [
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.PPMI),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.HAL_cwr),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.LLR),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.LLR_Z),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.LLR_N),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.DICE),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.TF_IDF),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.TF_normalized),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.PPMI),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.HAL_cwr),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.LLR),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.LLR_Z),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.LLR_N),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.DICE),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.TF_IDF),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.TF_normalized),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Weighed, KeynessMetric.PPMI),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Weighed, KeynessMetric.HAL_cwr),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Weighed, KeynessMetric.LLR),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Weighed, KeynessMetric.LLR_Z),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Weighed, KeynessMetric.LLR_N),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Weighed, KeynessMetric.DICE),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Weighed, KeynessMetric.TF_IDF),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Weighed, KeynessMetric.TF_normalized),
    ],
)
def test_keyness_transform_corpus(tag: str, keyness_source: KeynessMetricSource, keyness: KeynessMetric):
    folder: str = f'./tests/test_data/{tag}'
    bundle: Bundle = Bundle.load(folder=folder, tag=tag, compute_frame=False)
    opts: ComputeKeynessOpts = ComputeKeynessOpts(
        period_pivot="year",
        keyness_source=keyness_source,
        keyness=keyness,
        tf_threshold=1,
        pivot_column_name='time_period',
        normalize=False,
        fill_gaps=False,
    )
    corpus: VectorizedCorpus = bundle.keyness_transform(opts=opts)

    assert corpus is not None


@pytest.mark.parametrize(
    "tag,keyness_source,keyness",
    [
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.HAL_cwr),
    ],
)
def test_keyness_transform_corpus2(tag: str, keyness_source: KeynessMetricSource, keyness: KeynessMetric):
    folder: str = f'./tests/test_data/{tag}'
    bundle: Bundle = Bundle.load(folder=folder, tag=tag, compute_frame=False)
    opts: ComputeKeynessOpts = ComputeKeynessOpts(
        period_pivot="year",
        keyness_source=keyness_source,
        keyness=keyness,
        tf_threshold=10,
        pivot_column_name='time_period',
        normalize=False,
        fill_gaps=False,
    )
    corpus: VectorizedCorpus = bundle.keyness_transform(opts=opts)

    assert corpus is not None


def test_zero_out_by_tf_threshold():
    expected_sums = [28, 12, 9, 11, 39, 34, 7, 8, 15, 16, 10, 34, 8, 28, 14, 19, 28, 23, 23, 9, 16, 9, 16, 4, 16, 17, 4]
    tag: str = 'ABCDEFG_7DOCS_CONCEPT'
    folder: str = f'./tests/test_data/{tag}'
    bundle: Bundle = Bundle.load(folder=folder, tag=tag, compute_frame=False)

    corpus: VectorizedCorpus = bundle.corpus

    assert (corpus.term_frequency == expected_sums).all()

    tf_threshold: int = 10
    indices = [i for i, v in enumerate(expected_sums) if v < tf_threshold]
    for i in indices:
        expected_sums[i] = 0

    corpus.zero_out_by_tf_threshold(tf_threshold)

    assert (corpus.term_frequency == expected_sums).all()
