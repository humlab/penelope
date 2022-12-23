from penelope.co_occurrence import Bundle, to_filename
from penelope.co_occurrence.keyness import ComputeKeynessOpts
from penelope.common.keyness import KeynessMetric, KeynessMetricSource

# pylint: disable=protected-access, redefined-outer-name


def load_bundle(folder: str, tag: str):
    filename = to_filename(folder=folder, tag=tag)
    bundle: Bundle = Bundle.load(filename, compute_frame=False)
    return bundle


def compute():

    folder, tag = (
        '/path/to/data',
        'data-tag',
    )
    bundle: Bundle = load_bundle(folder, tag)

    opts: ComputeKeynessOpts = ComputeKeynessOpts(
        keyness_source=KeynessMetricSource.Weighed,
        keyness=KeynessMetric.LLR,
        tf_threshold=1,
        pivot_column_name="time_period",
        temporal_pivot="decade",
        fill_gaps=False,
        normalize=False,
    )

    bundle.keyness_transform(opts=opts)


if __name__ == '__main__':
    compute()
