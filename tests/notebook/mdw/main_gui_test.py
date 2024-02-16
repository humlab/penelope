import pytest

import penelope.notebook.mdw as mdw
from penelope.notebook import pick_file_gui as pfg
from tests.utils import TEST_DATA_FOLDER

from ...utils import create_abc_corpus

# pylint: disable=unused-argument, redefined-outer-name, protected-access


@pytest.fixture
def corpus_fixture():
    return create_abc_corpus(
        dtm=[
            [2, 1, 4, 1],
            [2, 2, 3, 0],
            [2, 3, 2, 0],
            [2, 4, 1, 1],
            [2, 0, 1, 1],
        ],
        document_years=[2013, 2013, 2014, 2014, 2014],
    )


def monkey_patch(*_, **__): ...


def test_create_main_gui(corpus_fixture):
    gui: pfg.PickFileGUI = mdw.create_main_gui(folder=TEST_DATA_FOLDER)
    assert isinstance(gui, pfg.PickFileGUI)
    assert isinstance(gui.payload, mdw.MDW_GUI)

    gui.payload.corpus = corpus_fixture
    gui.payload._period1.value = (2013, 2013)
    gui.payload._period2.value = (2014, 2014)
    gui.payload._top_n_terms.value = 2
    gui.payload._compute.click()
