# import os

# import pytest

# from penelope import co_occurrence
# from penelope.common.keyness import KeynessMetric
# from penelope.notebook import utility as notebook_utility
# from penelope.notebook.co_occurrence import ExploreGUI
# from penelope.notebook.word_trends import BundleTrendsData, TrendsBaseGUI, TrendsComputeOpts

# # pylint: disable=redefined-outer-name

# TEST_BUNDLE: str = "/data/inidun/courier/co_occurrence/courier_issue_by_article_pages_20221220/Courier_civilizations_w5_tf10_nvadj_nolemma"

# # @pytest.fixture(scope="module")
# # def bundle() -> co_occurrence.Bundle:
# #     folder, tag = TEST_BUNDLE, os.path.split(TEST_BUNDLE)[1]
# #     filename = co_occurrence.to_filename(folder=folder, tag=tag)
# #     bundle: co_occurrence.Bundle = co_occurrence.Bundle.load(filename, compute_frame=False)
# #     return bundle


# # @pytest.fixture(scope="module")
# # def trends_data(bundle) -> BundleTrendsData:
# #     trends_data: BundleTrendsData = BundleTrendsData(bundle=bundle)
# #     return trends_data


# def test_load_bundle():

#     folder, tag = TEST_BUNDLE, os.path.split(TEST_BUNDLE)[1]
#     filename = co_occurrence.to_filename(folder=folder, tag=tag)

#     bundle: co_occurrence.Bundle = co_occurrence.Bundle.load(filename, compute_frame=False)

#     assert bundle is not None
#     bundle.compress(tf_threshold=1)

#     # NOTE! Must add magic ids to call

#     gui_explore: ExploreGUI = ExploreGUI(bundle=bundle)
#     gui_explore.setup()
#     gui_explore.layout()

#     trends_gui: TrendsBaseGUI = gui_explore.trends_gui

#     trends_data: BundleTrendsData = BundleTrendsData(bundle=bundle)

#     # gui_explore.display(trends_data=trends_data)
#     #  --> trends_data.transform(compute_opts)
#     #  --> gui_explore.trends_gui.display(trends_data=trends_data)

#     compute_opts: TrendsComputeOpts = TrendsComputeOpts(
#         normalize=False,
#         smooth=False,
#         keyness=KeynessMetric.TF,
#         temporal_key="year",  # "decade", "lustrum"
#         top_count=100,
#         words="*",
#         descending=True,
#     )

#     trends_data.transform(compute_opts)

#     trends_gui.plot()


# # def test_ExploreCoOccurrencesGUI_create_and_layout(bundle: co_occurrence.Bundle):

# #     gui: ExploreGUI = ExploreGUI(bundle=bundle).setup()

# #     _: notebook_utility.OutputsTabExt = gui.layout()


# # @pytest.mark.long_running
# # def test_ExploreCoOccurrencesGUI_display(bundle: co_occurrence.Bundle, trends_data: BundleTrendsData):

# #     gui: ExploreGUI = ExploreGUI(bundle=bundle).setup()

# #     _: notebook_utility.OutputsTabExt = gui.layout()

# #     gui.display(trends_data)
