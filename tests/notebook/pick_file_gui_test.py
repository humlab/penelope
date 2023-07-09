from typing import Any
from unittest import mock

import pytest

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


@pytest.mark.parametrize('kind', ['chooser', 'picker'])
def test_create_load_gui(kind):
    folder = './tests/test_data'
    is_called: bool = False

    def picked_callback(filename: str, sender: Any) -> None:  # pylint: disable=unused-argument
        nonlocal is_called
        is_called = True

    with mock.patch(
        'penelope.notebook.pick_file_gui.PickFileGUI.filename', new_callable=mock.PropertyMock
    ) as mocked_corpus_filename:
        mocked_corpus_filename.return_value = "./tests/"

        gui = PickFileGUI(folder=folder, pattern='*_vector_data.npz', picked_callback=picked_callback, kind=kind)
        gui.setup()
        gui.load()

        assert is_called
        assert gui._alert.value == "<span style='color: green; font-weight: bold;'>✔</span>"
