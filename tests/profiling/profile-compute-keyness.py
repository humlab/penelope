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
        '/home/roger/source/welfare-state-analytics/welfare_state_analytics/data/1970-information',
        '1970-information',
    )
    bundle: Bundle = load_bundle(folder, tag)

    opts: ComputeKeynessOpts = ComputeKeynessOpts(
        keyness_source=KeynessMetricSource.Weighed,
        keyness=KeynessMetric.LLR,
        tf_threshold=1,
        pivot_column_name="time_period",
        period_pivot="decade",  # ["year", "lustrum", "decade"],
        fill_gaps=False,
        normalize=False,
    )

    bundle.keyness_transform(opts=opts)

    # gui: TabularCoOccurrenceGUI = TabularCoOccurrenceGUI(bundle=bundle)
    # gui.stop_observe()
    # gui.pivot = time_period
    # gui.keyness_source = KeynessMetricSource.Full
    # gui.keyness = KeynessMetric.TF
    # gui.token_filter = "educational/*"
    # gui.global_threshold = 50
    # gui.concepts = set(["general"])
    # gui.largest = 10
    # gui.start_observe()

    # gui.update_corpus()
    # gui.update_co_occurrences()


if __name__ == '__main__':
    compute()
