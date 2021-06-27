import pytest
from penelope.co_occurrence import Bundle, ContextOpts
from penelope.co_occurrence.keyness import ComputeKeynessOpts
from penelope.common.keyness import KeynessMetric, KeynessMetricSource
from penelope.corpus import VectorizedCorpus

from .utils import create_simple_bundle_by_pipeline


SIMPLE_CORPUS_ABCDE_3DOCS = [
    ('tran_2019_01_test.txt', ['a', 'b', 'c', 'c', 'd', 'c', 'e']),
    ('tran_2019_02_test.txt', ['a', 'a', 'c', 'e', 'c', 'd', 'd']),
    ('tran_2019_03_test.txt', ['d', 'e', 'e', 'b']),
]

def test_keyness_transform_with_simple_corpus():
    context_opts: ContextOpts = ContextOpts(concept={'d'}, ignore_concept=False, context_width=1)
    bundle: Bundle = create_simple_bundle_by_pipeline(data=SIMPLE_CORPUS_ABCDE_3DOCS, context_opts=context_opts)
    keyness_source: KeynessMetricSource = KeynessMetricSource.Weighed
    keyness: KeynessMetric = KeynessMetric.TF_normalized
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
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.PPMI),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.HAL_cwr),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.LLR),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.LLR_Dunning),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.DICE),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.TF_IDF),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.TF_normalized),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.PPMI),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.HAL_cwr),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.LLR),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.LLR_Dunning),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.DICE),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.TF_IDF),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.TF_normalized),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Weighed, KeynessMetric.PPMI),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Weighed, KeynessMetric.HAL_cwr),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Weighed, KeynessMetric.LLR),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Weighed, KeynessMetric.LLR_Dunning),
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
