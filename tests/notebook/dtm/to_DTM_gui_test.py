from penelope.notebook import dtm
from penelope.pipeline import CorpusConfig


def dummy_callback(*_, **__):
    pass


def dummy_config():
    return CorpusConfig.load(path='./tests/test_data/SSI/SSI.yml')


def test_GUI_setup_and_layout():  # pylint: disable=unused-argument
    def done_callback(*_, **__):
        pass

    def compute_callback(args, corpus_config):  # pylint: disable=unused-argument
        pass

    corpus_config = dummy_config()
    gui = dtm.ComputeGUI(
        default_corpus_path='./tests/test_data',
        default_corpus_filename='',
        default_data_folder='./tests/output',
    ).setup(
        config=corpus_config,
        compute_callback=compute_callback,
        done_callback=done_callback,
    )

    assert gui is not None
    assert gui.layout() is not None


def test_create_gui():
    gui: dtm.ComputeGUI = dtm.create_compute_gui(
        corpus_folder='./tests/test_data',
        data_folder='./tests/test_data',
        config=dummy_config(),
        compute_callback=dummy_callback,
        done_callback=dummy_callback,
    )

    assert gui is not None
