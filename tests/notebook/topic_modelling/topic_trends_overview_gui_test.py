

from penelope.notebook.topic_modelling.topic_trends_overview_gui import TopicOverviewGUI
from penelope.notebook.topic_modelling import TopicModelContainer


def test_create_gui(state: TopicModelContainer):
    gui: TopicOverviewGUI = TopicOverviewGUI()
    assert gui is not None

    gui = gui.setup(state)
    assert gui is not None

    layout = gui.layout()
    assert layout is not None

    gui.update_handler()

