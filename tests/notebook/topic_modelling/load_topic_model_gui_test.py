from typing import Callable, List

from penelope import pipeline
from penelope.notebook.topic_modelling.load_topic_model_gui import LoadGUI, create_load_topic_model_gui


def test_load_gui(state):
    corpus_folder: str = './tests/test_data'
    gui: LoadGUI = LoadGUI(
        corpus_folder=corpus_folder,
        state=state,
        corpus_config=None,
        slim=False,
    ).setup()

    layout = gui.layout()
    assert layout is not None

    gui.load()
