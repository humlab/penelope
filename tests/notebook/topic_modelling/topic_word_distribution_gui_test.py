from unittest import mock

from penelope.notebook.topic_modelling import TopicModelContainer
from penelope.notebook.topic_modelling.topic_word_distribution_gui import TopicWordDistributionGUI


@mock.patch('bokeh.plotting.show', lambda *_, **__: None)
@mock.patch('bokeh.io.push_notebook', lambda *_, **__: None)
def test_create_gui(state: TopicModelContainer):
    gui = TopicWordDistributionGUI(state)
    assert gui is not None

    gui = gui.setup()
    assert gui is not None

    layout = gui.layout()
    assert layout is not None

    gui.update_handler()
