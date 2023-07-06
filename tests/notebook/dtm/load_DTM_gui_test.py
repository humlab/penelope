# from unittest import mock

# from penelope.corpus import VectorizedCorpus, load_corpus
from penelope.notebook.dtm import LoadGUI

# pylint: disable=protected-access


def dummy_callback(*_, **__):
    pass


def test_gui_setup():
    gui = LoadGUI(folder='./tests/test_data', done_callback=None)
    assert gui.setup() is not None


def test_gui_layout():
    gui = LoadGUI(folder='./tests/test_data', done_callback=None)
    assert gui.setup().layout() is not None

