import pytest
from penelope.notebook.topic_modelling.display_topic_titles import display_gui
from penelope.notebook.topic_modelling.topics_token_network_gui import TopicsTokenNetworkGUI
from penelope.topic_modelling import InferredTopicsData

from .utility import load_inferred_topics_data

# pylint: disable=protected-access, redefined-outer-name


@pytest.fixture
def inferred_topics_data() -> InferredTopicsData:
    return load_inferred_topics_data()


def test_display_gui(inferred_topics_data: InferredTopicsData):
    display_gui(inferred_topics_data.topic_titles())

