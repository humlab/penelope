from unittest.mock import patch

import pandas as pd

from penelope.notebook.topic_modelling import TopicModelContainer
from penelope.notebook.topic_modelling.topic_topic_network_gui import TopicTopicGUI

# pylint: disable=protected-access

@patch('bokeh.plotting.show', lambda *_, **__: None)
def test_create_gui(state: TopicModelContainer):
    gui: TopicTopicGUI = TopicTopicGUI(state)

    assert gui is not None

    gui = gui.setup()

    gui.observe(False)

    gui._threshold.min = 0.0
    gui._threshold.value = 0.000001
    gui._n_docs.value = 1

    assert gui is not None

    layout = gui.layout()
    assert layout is not None

    data: pd.DataFrame = gui.update()

    assert data is not None
    assert len(data) > 0

    gui.update_handler()

    assert gui._alert.value.startswith("✅")
