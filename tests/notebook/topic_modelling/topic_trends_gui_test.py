from unittest import mock

from penelope import topic_modelling as tm
from penelope.notebook.topic_modelling import TopicModelContainer, topic_trends_gui


def monkey_patch(*_, **__):
    ...


def test_create_gui(state: TopicModelContainer):
    calculator: mock.MagicMock = mock.MagicMock(spec=tm.prevelance.TopicPrevalenceOverTimeCalculator)
    gui: topic_trends_gui.TopicTrendsGUI = topic_trends_gui.TopicTrendsGUI(calculator=calculator)
    assert gui is not None

    gui = gui.setup(state)
    assert gui is not None

    layout = gui.layout()
    assert layout is not None

    with mock.patch(
        'penelope.notebook.topic_modelling.topic_trends_gui_utility.display_topic_trends',
        return_value='mockelimocked',
    ):
        gui.update_handler()
