from penelope.notebook.topic_modelling import TopicModelContainer
from penelope.notebook.topic_modelling.topic_documents_gui import TopicDocumentsGUI


def test_create_gui(state: TopicModelContainer):

    gui: TopicDocumentsGUI = TopicDocumentsGUI().setup(state=state)

    layout = gui.layout()
    assert layout is not None

    gui.goto_next()
    gui.goto_previous()
    gui.update_handler()
