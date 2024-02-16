from unittest import mock

import ipywidgets as widgets  # type: ignore
import pandas as pd
import pytest

from penelope.notebook.topic_modelling import TopicModelContainer
from penelope.notebook.topic_modelling import topic_document_network_gui as tdn_gui


# pylint: disable=protected-access, redefined-outer-name
def monkey_patch(*_, **__): ...


def test_create_gui(state: TopicModelContainer):
    ui: tdn_gui.TopicDocumentNetworkGui = tdn_gui.TopicDocumentNetworkGui(pivot_key_specs={}, state=state).setup()
    assert isinstance(ui, tdn_gui.TopicDocumentNetworkGui)

    layout = ui.layout()
    assert isinstance(layout, widgets.VBox)


def test_compile_data(state: TopicModelContainer):
    ui: tdn_gui.TopicDocumentNetworkGui = tdn_gui.TopicDocumentNetworkGui(pivot_key_specs={}, state=state).setup()
    ui._threshold.value = 0.5
    ui._year_range.value = (1900, 2100)
    ui._topic_ids.value = []
    ui._output_format.value = "table"
    ui._scale.value = 0.5
    ui._network_layout.value = tdn_gui.LAYOUT_OPTIONS[-1]
    data: pd.DataFrame = ui.update()
    assert data is not None


@pytest.mark.parametrize('cls_name', [tdn_gui.DefaultTopicDocumentNetworkGui, tdn_gui.FocusTopicDocumentNetworkGui])
@mock.patch('penelope.notebook.grid_utility.table_widget')
@mock.patch('bokeh.plotting.show')
def test_display_document_topic_network(
    mock_show: mock.Mock, mock_display: mock.Mock, cls_name: tdn_gui.TopicDocumentNetworkGui, state: TopicModelContainer
):
    pivot_key_specs: dict = {}
    ui: tdn_gui.TopicDocumentNetworkGui = cls_name(pivot_key_specs=pivot_key_specs, state=state).setup()

    _ = ui.layout()

    ui._output_format.value = "table"
    ui.update()
    ui.display_handler()
    # assert mock_display.called and not mock_show.called

    mock_display.reset_mock()
    mock_show.reset_mock()

    ui._output_format.value = "network"
    ui.update()
    ui.display_handler()
    # assert mock_show.called and not mock_display.called
