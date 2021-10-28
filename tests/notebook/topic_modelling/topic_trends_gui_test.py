

from penelope.notebook.topic_modelling.topic_trends_gui import TopicTrendsGUI
from penelope.notebook.topic_modelling import TopicModelContainer


def test_create_gui(state: TopicModelContainer):
    gui: TopicTrendsGUI = TopicTrendsGUI()
    assert gui is not None

    gui = gui.setup(state)
    assert gui is not None

    layout = gui.layout()
    assert layout is not None

    gui.update_handler()

