from unittest import mock

import pandas as pd
import pytest
from ipywidgets import VBox

from penelope.corpus import dtm
from penelope.notebook.mdw import main_gui
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


def monkey_patch(*_, **__):
    ...


def test_create_main_gui():
    gui = main_gui.create_main_gui(folder=TEST_DATA_FOLDER, picked_callback=monkey_patch)

    assert isinstance(gui, VBox)


@mock.patch('IPython.display.display', monkey_patch)
@mock.patch('penelope.notebook.grid_utility.table_widget', monkey_patch)
def test_display_mdw():
    corpus = mock.MagicMock(spec=dtm.VectorizedCorpus)
    df_mdw = mock.MagicMock(spec=pd.DataFrame)
    main_gui.display_mdw(corpus, df_mdw)
