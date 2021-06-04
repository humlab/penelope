import pytest
from penelope import co_occurrence
from penelope.notebook import utility as notebook_utility
from penelope.notebook.co_occurrence import ExploreGUI, main_gui
from penelope.notebook.word_trends import TrendsData

# pylint: disable=redefined-outer-name


@pytest.fixture(scope="module")
def bundle() -> co_occurrence.Bundle:
    folder, tag = './tests/test_data/VENUS', 'VENUS'
    filename = co_occurrence.to_filename(folder=folder, tag=tag)
    bundle: co_occurrence.Bundle = co_occurrence.Bundle.load(filename, compute_frame=False)
    return bundle


@pytest.fixture(scope="module")
def trends_data(bundle) -> TrendsData:
    trends_data: TrendsData = main_gui.to_trends_data(bundle).update()
    return trends_data


def test_ExploreCoOccurrencesGUI_create_and_layout(bundle: co_occurrence.Bundle):

    gui: ExploreGUI = ExploreGUI(bundle=bundle).setup()

    _: notebook_utility.OutputsTabExt = gui.layout()


def test_ExploreCoOccurrencesGUI_display(bundle: co_occurrence.Bundle, trends_data: TrendsData):

    gui: ExploreGUI = ExploreGUI(bundle=bundle).setup()

    _: notebook_utility.OutputsTabExt = gui.layout()

    gui.display(trends_data)


# @pytest.fixture(scope="module")
# def apa() -> co_occurrence.Bundle:
#     folder, tag = '/data/westac/data/APA', 'APA'
#     filename = co_occurrence.to_filename(folder=folder, tag=tag)
#     bundle: co_occurrence.Bundle = co_occurrence.Bundle.load(filename, compute_frame=False)
#     return bundle


# def test_load_apa(apa: co_occurrence.Bundle):

#     corpus: VectorizedCorpus = apa.corpus

#     gui: ExploreGUI = ExploreGUI(bundle=apa).setup()

#     assert gui is not None

#     trends_data: TrendsData = TrendsData().update(
#         corpus=corpus,
#         corpus_folder=apa.folder,
#         corpus_tag=apa.tag,
#         n_count=100000,
#     )

#     assert trends_data is not None
