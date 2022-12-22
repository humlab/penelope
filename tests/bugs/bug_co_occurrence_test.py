import os

from penelope import co_occurrence
from penelope.common.keyness import KeynessMetric
from penelope.notebook.co_occurrence import ExploreGUI
from penelope.notebook.word_trends import BundleTrendsData, TrendsBaseGUI, TrendsComputeOpts

# pylint: disable=redefined-outer-name

TEST_BUNDLE: str = "/home/roger/source/penelope/opts/inidun/co_occurrence/courier_issue_by_article_pages_20221220/data/Courier_civilizations_w5_tf10_nvadj_nolemma"


def test_load_bundle():

    folder, tag = TEST_BUNDLE, os.path.split(TEST_BUNDLE)[1]
    filename = co_occurrence.to_filename(folder=folder, tag=tag)

    bundle: co_occurrence.Bundle = co_occurrence.Bundle.load(filename, compute_frame=False)

    assert bundle is not None
    bundle.compress(tf_threshold=1)

    # NOTE! Must add magic ids to call

    gui_explore: ExploreGUI = ExploreGUI(bundle=bundle)
    gui_explore.setup()
    gui_explore.layout()

    trends_gui: TrendsBaseGUI = gui_explore.trends_gui

    trends_data: BundleTrendsData = BundleTrendsData(bundle=bundle)

    # gui_explore.display(trends_data=trends_data)
    #  --> trends_data.transform(compute_opts)
    #  --> gui_explore.trends_gui.display(trends_data=trends_data)

    compute_opts: TrendsComputeOpts = TrendsComputeOpts(
        normalize=False,
        smooth=False,
        keyness=KeynessMetric.TF,
        temporal_key="year",  # "decade", "lustrum"
        top_count=100,
        words="*",
        descending=True,
    )

    trends_data.transform(compute_opts)

    trends_gui.plot()
