# from unittest import mock

# from penelope.corpus import VectorizedCorpus, load_corpus
from penelope.notebook.pick_file_gui import PickFileGUI

# pylint: disable=protected-access


def dummy_callback(*_, **__):
    pass


def test_gui_setup():
    gui = PickFileGUI(folder='./tests/test_data', pattern='*.*', picked_callback=None)
    assert gui.setup() is not None


def test_gui_layout():
    gui = PickFileGUI(folder='./tests/test_data', pattern='*.*', picked_callback=None)
    assert gui.setup().layout() is not None
