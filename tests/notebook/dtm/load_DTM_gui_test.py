from penelope.notebook.dtm import LoadGUI, create_load_gui


def dummy_callback(*_, **__):
    pass


def test_gui_setup():
    gui = LoadGUI(
        default_corpus_folder='./tests/test_data', filename_pattern='*.*', load_callback=None, done_callback=None
    )
    assert gui.setup() is not None


def test_gui_layout():
    gui = LoadGUI(
        default_corpus_folder='./tests/test_data', filename_pattern='*.*', load_callback=None, done_callback=None
    )
    assert gui.setup().layout() is not None


def test_create_gui():

    gui = create_load_gui(
        corpus_folder='./tests/test_data',
        loaded_callback=dummy_callback,
    )

    assert gui is not None
