from unittest import mock

import ipywidgets as widgets
import pandas as pd
import pytest
from penelope.notebook.topic_modelling import topic_document_network_gui as tdn_gui
from penelope.topic_modelling import InferredTopicsData
from tests.utils import PERSISTED_INFERRED_MODEL_SOURCE_FOLDER


# pylint: disable=protected-access, redefined-outer-name
def monkey_patch(*_, **__):
    ...


# @pytest.fixture
# def inferred_model() -> InferredModel:
#     _inferred_model: InferredModel = topic_modelling.load_model(PERSISTED_INFERRED_MODEL_SOURCE_FOLDER)
#     return _inferred_model


@pytest.fixture
def inferred_topics_data() -> InferredTopicsData:
    # _inferred_topics_data: InferredTopicsData = topic_modelling.compile_inferred_topics_data(
    #     topic_model=inferred_model.topic_model,
    #     corpus=inferred_model.train_corpus.corpus,
    #     id2word=inferred_model.train_corpus.id2word,
    #     document_index=inferred_model.train_corpus.document_index,
    #     n_tokens=5,
    # )
    # _inferred_topics_data.store(target_folder=PERSISTED_INFERRED_MODEL_SOURCE_FOLDER, pickled=False)
    filename_fields = ["year:_:1", "year_serial_id:_:2"]
    _inferred_topics_data = InferredTopicsData.load(
        folder=PERSISTED_INFERRED_MODEL_SOURCE_FOLDER, filename_fields=filename_fields
    )
    return _inferred_topics_data


def test_compile_data(inferred_topics_data):

    opts: tdn_gui.GUI.GUI_opts = tdn_gui.GUI.GUI_opts(
        plot_mode=tdn_gui.PlotMode.Default,
        inferred_topics=inferred_topics_data,
        threshold=0.5,
        period=(1900, 2100),
        topic_ids=None,
        output_format="table",
        scale=0.5,
        layout_algorithm=tdn_gui.NETWORK_LAYOUT_ALGORITHMS[-1],
    )
    data: pd.DataFrame = tdn_gui.compile_network_data(opts=opts)

    assert data is not None


def test_topic_document_network_gui(inferred_topics_data: InferredTopicsData):

    plot_mode = tdn_gui.PlotMode.Default

    gui: tdn_gui.GUI = tdn_gui.GUI(plot_mode=plot_mode).setup(inferred_topics=inferred_topics_data)
    assert isinstance(gui, tdn_gui.GUI)

    layout = gui.layout()
    assert isinstance(layout, widgets.VBox)

    opts = gui.opts
    assert isinstance(opts, tdn_gui.GUI.GUI_opts)


@pytest.mark.parametrize('plot_mode', [tdn_gui.PlotMode.Default, tdn_gui.PlotMode.FocusTopics])
@mock.patch('IPython.display.display')  # , monkey_patch)
@mock.patch('bokeh.plotting.show')  # , monkey_patch)
def test_display_document_topic_network(
    mock_show: mock.Mock, mock_display: mock.Mock, plot_mode, inferred_topics_data: InferredTopicsData
):

    plot_mode = tdn_gui.PlotMode.Default

    gui: tdn_gui.GUI = tdn_gui.GUI(plot_mode=plot_mode).setup(inferred_topics=inferred_topics_data)

    _ = gui.layout()

    opts = gui.opts
    opts.plot_mode = plot_mode

    opts.output_format = "table"
    tdn_gui.display_document_topic_network(opts)
    assert mock_display.called and not mock_show.called

    mock_display.reset_mock()
    mock_show.reset_mock()

    opts.output_format = "network"
    tdn_gui.display_document_topic_network(opts)
    assert mock_show.called and not mock_display.called


# # def display_gui(plot_mode:PlotMode.FocusTopics, state: TopicModelContainer):

# #     gui: GUI = GUI().setup(plot_mode, state.inferred_topics, state.num_topics)
# #     display(gui.layout())
# #     gui.update_handler()
