from penelope.notebook.topic_modelling.topic_topic_network_gui import TopicTopicGUI
from penelope.notebook.topic_modelling import TopicModelContainer


def test_create_gui(state: TopicModelContainer):
    gui: TopicTopicGUI = TopicTopicGUI(state)
    assert gui is not None

    gui = gui.setup()
    assert gui is not None

    layout = gui.layout()
    assert layout is not None

    gui.update_handler()
