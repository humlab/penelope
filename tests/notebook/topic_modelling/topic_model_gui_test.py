from penelope.notebook.topic_modelling.topic_model_gui import ModelWidgetsGUI, PreparedCorpusGUI
from penelope.notebook.topic_modelling import TopicModelContainer


def test_create_gui(state: TopicModelContainer):
    gui: ModelWidgetsGUI = ModelWidgetsGUI()
    assert gui is not None

    callback = lambda terms, document_index, opts: None
    gui.setup(callback)

    # data_folder='./tests/test_data', state=state, fn_doc_index=lambda **_: _ )

    # gui = gui.setup()
    # assert gui is not None

    # layout = gui.layout()
    # assert layout is not None

    # gui.update_handler()
