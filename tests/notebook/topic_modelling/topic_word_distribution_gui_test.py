from penelope.notebook.topic_modelling import TopicModelContainer
from penelope.notebook.topic_modelling.topic_word_distribution_gui import TopicWordDistributionGUI


def test_create_gui(state: TopicModelContainer):
    gui = TopicWordDistributionGUI(state)
    assert gui is not None

    gui = gui.setup()
    assert gui is not None

    layout = gui.layout()
    assert layout is not None

    gui.tick()

    gui.update_handler()
