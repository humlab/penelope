import pytest
from penelope.notebook.dtm.load_DTM_gui import LoadGUI, create_gui


def dummy_callback(*_, **__):
    pass


@pytest.mark.skip('')
def test_gui_create():
    assert (
        LoadGUI(default_corpus_folder='./tests/test_data', filename_pattern='*.*', load_callback=None).layout()
        is not None
    )


@pytest.mark.skip('')
def test_gui_setup():
    assert (
        LoadGUI(default_corpus_folder='./tests/test_data', filename_pattern='*.*', load_callback=None).setup()
        is not None
    )


@pytest.mark.skip('')
def test_create_gui():

    gui = create_gui(
        corpus_folder='./tests/test_data',
        loaded_callback=dummy_callback,
    )

    assert gui is not None
