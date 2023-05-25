import pytest

from penelope import co_occurrence
from penelope.notebook import utility as notebook_utility
from penelope.notebook.co_occurrence import ExploreGUI
from penelope.notebook.word_trends import BundleTrendsService

# pylint: disable=redefined-outer-name


@pytest.fixture(scope="module")
def bundle() -> co_occurrence.Bundle:
    folder, tag = './tests/test_data/VENUS', 'VENUS'
    filename = co_occurrence.to_filename(folder=folder, tag=tag)
    bundle: co_occurrence.Bundle = co_occurrence.Bundle.load(filename, compute_frame=False)
    return bundle


@pytest.fixture(scope="module")
def trends_data(bundle) -> BundleTrendsService:
    trends_data: BundleTrendsService = BundleTrendsService(bundle=bundle)
    return trends_data


def test_ExploreCoOccurrencesGUI_create_and_layout(bundle: co_occurrence.Bundle):
    gui: ExploreGUI = ExploreGUI(bundle=bundle).setup()

    _: notebook_utility.OutputsTabExt = gui.layout()


@pytest.mark.long_running
def test_ExploreCoOccurrencesGUI_display(bundle: co_occurrence.Bundle, trends_data: BundleTrendsService):
    gui: ExploreGUI = ExploreGUI(bundle=bundle).setup()

    _: notebook_utility.OutputsTabExt = gui.layout()

    gui.display(trends_data)
