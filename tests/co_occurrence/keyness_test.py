import pytest
from penelope.co_occurrence import Bundle, ContextOpts
from penelope.co_occurrence.keyness import ComputeKeynessOpts
from penelope.common.keyness import KeynessMetric, KeynessMetricSource
from penelope.corpus import VectorizedCorpus
from tests.fixtures import SIMPLE_CORPUS_ABCDEFG_7DOCS

from .utils import create_simple_bundle_by_pipeline


def test_keyness_transform_with_simple_corpus():
    # FIXME: Check compute when concept does not exist in corpys!!
    context_opts: ContextOpts = ContextOpts(concept={'g'}, ignore_concept=False, context_width=2)
    bundle: Bundle = create_simple_bundle_by_pipeline(data=SIMPLE_CORPUS_ABCDEFG_7DOCS, context_opts=context_opts)
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
        ('SSI', KeynessMetricSource.Full, KeynessMetric.PPMI),
        ('SSI', KeynessMetricSource.Full, KeynessMetric.HAL_cwr),
        ('SSI', KeynessMetricSource.Full, KeynessMetric.LLR),
        ('SSI', KeynessMetricSource.Full, KeynessMetric.LLR_Dunning),
        ('SSI', KeynessMetricSource.Full, KeynessMetric.DICE),
        ('SSI', KeynessMetricSource.Full, KeynessMetric.TF_IDF),
        ('SSI', KeynessMetricSource.Full, KeynessMetric.TF_normalized),
        ('SSI', KeynessMetricSource.Concept, KeynessMetric.PPMI),
        ('SSI', KeynessMetricSource.Concept, KeynessMetric.HAL_cwr),
        ('SSI', KeynessMetricSource.Concept, KeynessMetric.LLR),
        ('SSI', KeynessMetricSource.Concept, KeynessMetric.LLR_Dunning),
        ('SSI', KeynessMetricSource.Concept, KeynessMetric.DICE),
        ('SSI', KeynessMetricSource.Concept, KeynessMetric.TF_IDF),
        ('SSI', KeynessMetricSource.Concept, KeynessMetric.TF_normalized),
        ('SSI', KeynessMetricSource.Weighed, KeynessMetric.PPMI),
        ('SSI', KeynessMetricSource.Weighed, KeynessMetric.HAL_cwr),
        ('SSI', KeynessMetricSource.Weighed, KeynessMetric.LLR),
        ('SSI', KeynessMetricSource.Weighed, KeynessMetric.LLR_Dunning),
        ('SSI', KeynessMetricSource.Weighed, KeynessMetric.DICE),
        ('SSI', KeynessMetricSource.Weighed, KeynessMetric.TF_IDF),
        ('SSI', KeynessMetricSource.Weighed, KeynessMetric.TF_normalized),
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
