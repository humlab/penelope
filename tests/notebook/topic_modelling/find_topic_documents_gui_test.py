from penelope.notebook.topic_modelling.find_topic_documents_gui import FindTopicDocumentsGUI, create_gui


def test_load_gui(state):
    gui: FindTopicDocumentsGUI = FindTopicDocumentsGUI(state).setup()

    layout = gui.layout()
    assert layout is not None

    gui.update_handler()


def test_create_gui(state):

    gui: FindTopicDocumentsGUI = create_gui(state)

    layout = gui.layout()
    assert layout is not None

    gui.update_handler()
