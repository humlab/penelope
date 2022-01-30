from penelope.notebook.topic_modelling.topic_documents_gui import FindTopicDocumentsGUI


def test_load_gui(state):
    gui: FindTopicDocumentsGUI = FindTopicDocumentsGUI(state=state).setup()

    layout = gui.layout()
    assert layout is not None

    gui.update_handler()

