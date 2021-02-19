from unittest.mock import MagicMock, patch

import pandas as pd
import penelope.notebook.mdw.main_gui as main_gui
from ipywidgets import VBox
from penelope.corpus import dtm
from tests.utils import TEST_DATA_FOLDER


def monkey_patch(*_, **__):
    ...


def test_create_main_gui():

    gui = main_gui.create_main_gui(corpus_folder=TEST_DATA_FOLDER, loaded_callback=monkey_patch)

    assert isinstance(gui, VBox)


@patch('IPython.display.display', monkey_patch)
@patch('penelope.notebook.ipyaggrid_utility.display_grid', monkey_patch)
def test_display_mdw():
    corpus = MagicMock(spec=dtm.VectorizedCorpus)
    df_mdw = MagicMock(spec=pd.DataFrame)
    main_gui.display_mdw(corpus, df_mdw)


@patch('IPython.display.display', monkey_patch)
def test_default_loaded_callback():
    corpus = MagicMock(spec=dtm.VectorizedCorpus)
    corpus_folder = "./tests/test_data"
    corpus_tag = "dummpy"
    main_gui.default_loaded_callback(corpus, corpus_folder, corpus_tag)
