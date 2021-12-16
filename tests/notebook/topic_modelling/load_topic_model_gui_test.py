from typing import Callable, List

from penelope import pipeline
from penelope.notebook.topic_modelling.load_topic_model_gui import LoadGUI, create_load_topic_model_gui


def test_load_gui():
    model_names: List[str] = ["A", "B", "C"]
    load_callback: Callable = lambda _: _
    gui: LoadGUI = LoadGUI().setup(model_names, load_callback)

    layout = gui.layout()
    assert layout is not None

    gui.load_handler()


def test_create_load_topic_model_gui(state):

    corpus_config: pipeline.CorpusConfig = pipeline.CorpusConfig.load('./tests/test_data/tranströmer.yml')
    corpus_folder: str = './tests/test_data'

    gui: LoadGUI = create_load_topic_model_gui(corpus_config, corpus_folder, state)

    layout = gui.layout()
    assert layout is not None

    gui.load_handler()
