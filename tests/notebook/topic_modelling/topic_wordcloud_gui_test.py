from penelope.notebook.topic_modelling import TopicModelContainer
from penelope.notebook.topic_modelling.topic_wordcloud_gui import WordcloudGUI


def test_create_gui(state: TopicModelContainer):
    gui = WordcloudGUI(state=state)
    assert gui is not None

    gui = gui.setup()
    assert gui is not None

    layout = gui.layout()
    assert layout is not None

    gui.tick()

    gui.update_handler()
