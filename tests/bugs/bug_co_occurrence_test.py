import os

import pytest

from penelope import co_occurrence
from penelope.common.keyness import KeynessMetric, KeynessMetricSource
from penelope.notebook.co_occurrence import ExploreGUI
from penelope.notebook.word_trends import BundleTrendsService, CoOccurrenceTrendsGUI, TrendsComputeOpts

# pylint: disable=redefined-outer-name, protected-access

TEST_BUNDLE: str = (
    "/data/inidun/courier/co_occurrence/courier_issue_by_article_pages_20221220/Courier_civilizations_w5_tf10_nvadj_nolemma"
)


@pytest.mark.skip(reason="Bug fixed")
def test_load_bundle():
    folder, tag = TEST_BUNDLE, os.path.split(TEST_BUNDLE)[1]
    filename = co_occurrence.to_filename(folder=folder, tag=tag)

    bundle: co_occurrence.Bundle = co_occurrence.Bundle.load(filename, compute_frame=False)

    gui_explore: ExploreGUI = ExploreGUI(bundle=bundle)
    gui_explore.setup()
    gui_explore.layout()

    trends_service: BundleTrendsService = BundleTrendsService(bundle=bundle)

    gui_explore.display(trends_service)

    # compute_opts: TrendsComputeOpts = TrendsComputeOpts(
    #     normalize=False,
    #     smooth=False,
    #     keyness=KeynessMetric.TF,
    #     keyness_source=KeynessMetricSource.Weighed,
    #     temporal_key="decade",  # "decade", "lustrum"
    #     top_count=100,
    #     words="*",
    #     descending=True,
    #     fill_gaps=False
    # )

    compute_opts: TrendsComputeOpts = TrendsComputeOpts(
        normalize=False,
        keyness=KeynessMetric.TF,
        temporal_key='decade',
        pivot_keys_id_names=[],
        filter_opts=None,
        unstack_tabular=False,
        fill_gaps=False,
        smooth=False,
        top_count=5000,
        words='*',
        descending=True,
        keyness_source=KeynessMetricSource.Full,
    )

    trends_gui: CoOccurrenceTrendsGUI = gui_explore.trends_gui

    set_options(trends_gui, compute_opts)

    trends_gui.transform()

    assert len(trends_gui._words_picker.options) == 0

    # trends_gui._words_to_find.value = "*ment"
    trends_gui.update_picker()

    assert len(trends_gui._words_picker.options) > 0

    trends_gui.plot()
    assert trends_gui._alert.value == '🙃 Please specify tokens to plot'

    trends_gui._words_picker.value = trends_gui._words_picker.options[:1]
    assert trends_gui._alert.value == '🙂'

    # More isolated:
    # trends_gui.trends_service.transform(trends_gui.options)


def set_options(trends_gui: CoOccurrenceTrendsGUI, opts: TrendsComputeOpts) -> CoOccurrenceTrendsGUI:
    trends_gui.buzy(True)
    trends_gui.observe(False)
    trends_gui._smooth.value = opts.smooth
    trends_gui._normalize.value = opts.normalize
    trends_gui._keyness.value = opts.keyness
    trends_gui._keyness_source.value = opts.keyness_source
    trends_gui._temporal_key.value = opts.temporal_key
    trends_gui._top_words_count.value = opts.top_count
    trends_gui._words_to_find.value = opts.words
    trends_gui.observe(True)
    trends_gui.buzy(False)
    return trends_gui
