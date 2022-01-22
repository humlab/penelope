from unittest.mock import MagicMock, patch

import ipywidgets as widgets

import penelope.notebook.mdw.mdw_gui as mdw_gui
from penelope.corpus import dtm


def monkey_patch(*_, **__):
    ...


@patch('IPython.display.display', monkey_patch)
@patch('penelope.vendor.textacy.mdw_modified.compute_most_discriminating_terms', monkey_patch)
def test_default_compute_callback():
    corpus = MagicMock(spec=dtm.VectorizedCorpus)
    gui = MagicMock(spec=mdw_gui.MDW_GUI)
    mdw_gui.default_compute_callback(corpus, gui)


def test_mdw_gui():
    corpus = MagicMock(spec=dtm.VectorizedCorpus)
    gui = mdw_gui.MDW_GUI().setup(corpus, monkey_patch, monkey_patch)
    assert isinstance(gui, mdw_gui.MDW_GUI)
    layout = gui.layout()
    assert isinstance(layout, widgets.VBox)
    assert gui.period1 is not None
    assert gui.period2 is not None
    assert gui.top_n_terms is not None
    assert gui.max_n_terms is not None
    gui._compute_handler({})  # pylint: disable=protected-access


def test_create_mdw_gui():
    corpus = MagicMock(spec=dtm.VectorizedCorpus)

    gui = mdw_gui.create_mdw_gui(corpus, monkey_patch, monkey_patch)

    assert gui is not None
