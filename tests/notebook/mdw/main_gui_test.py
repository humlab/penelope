from unittest import mock

import pandas as pd
import pytest
from ipywidgets import VBox

from penelope.corpus import dtm
from penelope.notebook import dtm as dtm_ui
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
    gui = main_gui.create_main_gui(corpus_folder=TEST_DATA_FOLDER, loaded_callback=monkey_patch)

    assert isinstance(gui, VBox)


@mock.patch('IPython.display.display', monkey_patch)
@mock.patch('penelope.notebook.grid_utility.table_widget', monkey_patch)
def test_display_mdw():
    corpus = mock.MagicMock(spec=dtm.VectorizedCorpus)
    df_mdw = mock.MagicMock(spec=pd.DataFrame)
    main_gui.display_mdw(corpus, df_mdw)


def test_create_load_gui(corpus_fixture):
    corpus_folder = './tests/test_data'

    def display_patch(corpus: dtm.VectorizedCorpus):
        ...

    def load_corpus(folder: str, tag: str) -> dtm.VectorizedCorpus:
        return corpus_fixture

    with mock.patch(
        'penelope.notebook.dtm.LoadGUI.corpus_filename', new_callable=mock.PropertyMock
    ) as mocked_corpus_filename:
        mocked_corpus_filename.return_value = "./tests/"

        gui = dtm_ui.LoadGUI(
            folder=corpus_folder,
            filename_pattern="*.*",
            load_callback=load_corpus,
            done_callback=display_patch,
        ).setup()

        gui.is_dtm_corpus = mock.MagicMock(return_value=True)

        gui.load()

        assert gui._alert.value == "<span style='color=red'>✔</span>"
