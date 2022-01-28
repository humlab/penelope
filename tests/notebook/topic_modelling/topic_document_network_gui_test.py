from unittest import mock

import ipywidgets as widgets  # type: ignore
import pandas as pd
import pytest

from penelope.notebook.topic_modelling import TopicModelContainer
from penelope.notebook.topic_modelling import topic_document_network_gui as tdn_gui


# pylint: disable=protected-access, redefined-outer-name
def monkey_patch(*_, **__):
    ...


def test_create_gui(state: TopicModelContainer):

    plot_mode = tdn_gui.PlotMode.Default

    gui: tdn_gui.TopicDocumentNetworkGui = tdn_gui.TopicDocumentNetworkGui(state=state, plot_mode=plot_mode).setup()
    assert isinstance(gui, tdn_gui.TopicDocumentNetworkGui)

    layout = gui.layout()
    assert isinstance(layout, widgets.VBox)

    opts = gui.opts
    assert isinstance(opts, tdn_gui.TopicDocumentNetworkGui.GUI_opts)


def test_compile_data(state: TopicModelContainer):

    opts: tdn_gui.TopicDocumentNetworkGui.GUI_opts = tdn_gui.TopicDocumentNetworkGui.GUI_opts(
        plot_mode=tdn_gui.PlotMode.Default,
        inferred_topics=state["inferred_topics"],
        threshold=0.5,
        period=(1900, 2100),
        topic_ids=None,
        output_format="table",
        scale=0.5,
        layout_algorithm=tdn_gui.NETWORK_LAYOUT_ALGORITHMS[-1],
    )
    data: pd.DataFrame = tdn_gui.compile_network_data(opts=opts)

    assert data is not None


@pytest.mark.parametrize('plot_mode', [tdn_gui.PlotMode.Default, tdn_gui.PlotMode.FocusTopics])
@mock.patch('penelope.notebook.topic_modelling.topic_document_network_utility.display_document_topics_as_grid')
@mock.patch('bokeh.plotting.show')  # , monkey_patch)
def test_display_document_topic_network(
    mock_show: mock.Mock, mock_display: mock.Mock, plot_mode, state: TopicModelContainer
):

    plot_mode = tdn_gui.PlotMode.Default

    gui: tdn_gui.TopicDocumentNetworkGui = tdn_gui.TopicDocumentNetworkGui(state=state, plot_mode=plot_mode).setup()

    _ = gui.layout()

    opts = gui.opts
    opts.plot_mode = plot_mode

    opts.output_format = "table"
    tdn_gui.display_document_topic_network(opts)
    # assert mock_display.called and not mock_show.called

    mock_display.reset_mock()
    mock_show.reset_mock()

    opts.output_format = "network"
    tdn_gui.display_document_topic_network(opts)
    # assert mock_show.called and not mock_display.called
