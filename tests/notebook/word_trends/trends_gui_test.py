import unittest.mock as mock

import ipywidgets

from penelope.corpus import VectorizedCorpus
from penelope.notebook import pick_file_gui as pfg
from penelope.notebook.word_trends import ITrendDisplayer, TrendsGUI, TrendsService, main_gui

# from ..utils import create_abc_corpus
# pylint: disable=protected-access


def mocked_displayer_ctor(**_):
    m = mock.MagicMock(ITrendDisplayer)
    m.name = "apa"
    m.titles = "apa"
    return m


def test_TrendsGUI_setup():
    gui = TrendsGUI().setup(displayers=[mocked_displayer_ctor])
    assert len(gui._displayers) == 1  # pylint: disable=protected-access


def test_TrendsGUI_layout():
    w = TrendsGUI().setup(displayers=[mocked_displayer_ctor]).layout()
    assert isinstance(w, ipywidgets.CoreWidget)


def test_TrendsGUI_display():
    corpus = mock.Mock(spec=VectorizedCorpus)
    trends_service = mock.MagicMock(spec=TrendsService, corpus=corpus, category_column="apa")
    gui = TrendsGUI().setup(displayers=[mocked_displayer_ctor])
    gui.trends_service = trends_service
    gui.display(trends_service=trends_service)


def monkey_patch(*_, **__):
    ...


def test_create_simple_gui():
    folder: str = "tests/test_data/tranströmer/dtm"
    gui: pfg.PickFileGUI = main_gui.create_simple_dtm_gui(folder=folder)
    gui.setup()
    gui.layout()

    assert isinstance(gui, pfg.PickFileGUI)
    assert isinstance(gui.payload, TrendsGUI)
    assert isinstance(gui.payload.trends_service, TrendsService)

    assert gui.payload.trends_service.corpus is None

    gui._load_button.click()

    assert gui.payload.trends_service.corpus is not None


def test_advanced_gui():
    folder: str = "tests/test_data/tranströmer/dtm"

    gui: main_gui.ComplexTrendsGUI = main_gui.ComplexTrendsGUI(corpus_folder=folder, data_folder=folder)
    gui.setup()
    gui.layout()
    gui.gui_load._filename_picker.reset(path='tests/test_data/tranströmer/dtm', filename='tranströmer_vector_data.npz')
    gui.gui_load._load_button.click()

    assert gui.gui_load._alert.value == "<span style='color: green; font-weight: bold;'>✔</span>"

    assert gui.trends_service is not None
    assert gui.trends_service.corpus is not None
    # Add bad/good pivot keys to tranströmer corpus
    # assert gui.gui_trends.pivot_keys
    assert len(gui.content_placeholder.children) > 0
    assert gui.gui_trends._alert.value == "🥱 (not computed)"

    gui.gui_trends._compute.click()

    assert gui.gui_load._alert.value == "<span style='color: green; font-weight: bold;'>✔</span>"
    
    #assert gui.gui_trends._alert.value == "🙃 Please specify tokens to plot"
    #assert gui.gui_trends._alert.value == "🙂"

    # gui.gui_compute._config_chooser.reset(path='tests/test_data/tranströmer', filename='tranströmer.yml')
    # gui.gui_compute._config_chooser_changed()
    # gui.gui_compute._compute_button.click()
