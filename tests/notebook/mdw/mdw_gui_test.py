from unittest.mock import MagicMock, patch

import ipywidgets as widgets

import penelope.notebook.mdw.mdw_gui as mdw_gui
from penelope.corpus import dtm


def monkey_patch(*_, **__): ...


@patch('IPython.display.display', monkey_patch)
@patch('penelope.vendor.textacy_api._textacy.mdw_modified.compute_most_discriminating_terms', monkey_patch)
@patch('penelope.vendor.textacy_api.compute_most_discriminating_terms', monkey_patch)
def test_default_compute_callback():
    corpus = MagicMock(spec=dtm.VectorizedCorpus)
    gui = MagicMock(spec=mdw_gui.MDW_GUI)
    mdw_gui.default_compute_mdw(corpus, gui)


def test_mdw_gui():
    gui = mdw_gui.MDW_GUI().setup(monkey_patch)
    assert isinstance(gui, mdw_gui.MDW_GUI)
    layout = gui.layout()
    assert isinstance(layout, widgets.VBox)
    assert gui.period1 is not None
    assert gui.period2 is not None
    assert gui.top_n_terms is not None
    assert gui.max_n_terms is not None
    gui._compute_handler({})  # pylint: disable=protected-access
