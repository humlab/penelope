# Visualize year-to-topic network by means of topic-document-weights
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence, Tuple

import bokeh
import ipywidgets as widgets  # type: ignore
import networkx as nx
import pandas as pd
import penelope.network.plot_utility as network_plot
from IPython.display import display
from loguru import logger  # type: ignore
from penelope import topic_modelling, utility
from penelope.network.bipartite_plot import plot_bipartite_network
from penelope.network.networkx import utility as network_utility
from penelope.topic_modelling import InferredTopicsData

from .. import widgets_utils
from .model_container import TopicModelContainer
from .topic_document_network_utility import display_document_topics_as_grid

NETWORK_LAYOUT_ALGORITHMS = ["Circular", "Kamada-Kawai", "Fruchterman-Reingold"]


class PlotMode(IntEnum):
    Default = 1
    FocusTopics = 2


def display_document_topic_network(opts: "GUI.GUI_opts"):

    df_network: pd.DataFrame = compile_network_data(opts)

    if len(df_network) == 0:
        logger.info("No data")
        return

    df_network["title"] = df_network.filename

    if opts.output_format == "network":

        network: nx.Graph = network_utility.create_bipartite_network(df_network, "title", "topic_id")

        args = network_plot.layout_args(opts.layout_algorithm, network, opts.scale)

        layout_data = (network_plot.layout_algorithms[opts.layout_algorithm])(network, **args)

        titles: pd.DataFrame = topic_modelling.get_topic_titles(opts.inferred_topics.topic_token_weights)

        p = plot_bipartite_network(
            network,
            layout_data,
            scale=opts.scale,
            titles=titles,
            highlight_topic_ids=None if opts.plot_mode is None else opts.topic_ids,
            element_id=f"ID_{opts.plot_mode.name}",
        )

        bokeh.plotting.show(p)
    elif opts.output_format.lower() in ('xlsx', 'csv', 'clipboard'):
        utility.ts_store(data=df_network, extension=opts.output_format.lower(), basename='topic_topic_network')
    else:
        g = display_document_topics_as_grid(df_network)
        display(g)


def compile_network_data(opts: "GUI.GUI_opts") -> pd.DataFrame:

    document_topic_weights = opts.inferred_topics.document_topic_weights

    df_threshold: pd.DataFrame = document_topic_weights[document_topic_weights.weight > opts.threshold].reset_index()

    if len(opts.period or []) == 2:
        df_threshold = df_threshold[(df_threshold.year >= opts.period[0]) & (df_threshold.year <= opts.period[1])]

    if opts.plot_mode == PlotMode.FocusTopics:
        df_focus = df_threshold[df_threshold.topic_id.isin(opts.topic_ids)].set_index("document_id")
        df_others = df_threshold[~df_threshold.topic_id.isin(opts.topic_ids)].set_index("document_id")
        df_others = df_others[df_others.index.isin(df_focus.index)]
        df = df_focus.append(df_others).reset_index()
    else:
        df = (
            df_threshold[~df_threshold.topic_id.isin(opts.topic_ids)] if len(opts.topic_ids or []) > 0 else df_threshold
        )

    df["weight"] = utility.clamp_values(list(df.weight), (0.1, 2.0))

    if "filename" not in df:
        df = df.merge(
            opts.inferred_topics.document_index[['document_id', "filename"]].set_index('document_id'),
            left_on="document_id",
            right_index=True,
        )

    return df


class GUI:
    def __init__(self, plot_mode: PlotMode, inferred_topics: InferredTopicsData = None):

        self.plot_mode: PlotMode = plot_mode
        self.inferred_topics: InferredTopicsData = inferred_topics

        self.text: widgets.HTML = None
        self.period = widgets.IntRangeSlider(
            description="",
            min=1900,
            max=2030,
            step=1,
            value=(1900, 1900 + 5),
            continues_update=False,
        )
        self.scale = widgets.FloatSlider(
            description="",
            min=0.0,
            max=1.0,
            step=0.01,
            value=0.1,
            continues_update=False,
        )
        self.threshold = widgets.FloatSlider(
            description="",
            min=0.0,
            max=1.0,
            step=0.01,
            value=0.10,
            continues_update=False,
        )
        self.output_format = widgets.Dropdown(
            description="",
            options={"Network": "network", "Table": "table"},
            value="network",
            layout={'width': '200px'},
        )
        self.layout_algorithm = widgets.Dropdown(description="", options=[], value=None, layout={'width': '250px'})
        self.topic_ids = widgets.SelectMultiple(description="", options=[], value=[], rows=8, layout={'width': '180px'})
        self.button = widgets.Button(
            description="Display", button_style='Success', layout=widgets.Layout(width='115px', background_color='blue')
        )
        self.output = widgets.Output()

    def setup(self, inferred_topics: InferredTopicsData, default_threshold: float = None) -> "GUI":

        self.inferred_topics = inferred_topics
        self.threshold.value = default_threshold or (0.5 if self.plot_mode == PlotMode.Default else 0.1)
        self.topic_ids.options = ([("", None)] if self.plot_mode == PlotMode.Default else []) + [
            ("Topic #" + str(i), i) for i in range(0, inferred_topics.num_topics)
        ]
        self.topic_ids.value = []

        self.layout_algorithm.options = NETWORK_LAYOUT_ALGORITHMS
        self.layout_algorithm.value = self.layout_algorithm.options[-1]

        self.period.min, self.period.max = inferred_topics.year_period
        self.period.value = (self.period.min, self.period.min + 5)

        self.text = widgets_utils.text_widget(f"ID_{self.plot_mode.name}")
        self.button.on_click(self.update_handler)

        return self

    def update_handler(self, *_):
        self.output.clear_output()
        with self.output:
            display_document_topic_network(opts=self.opts)

    def layout(self):
        topics_specifier: str = "Ignore" if self.plot_mode == PlotMode.Default else "Focus"
        _layout = widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                widgets.HTML("<b>Year range</b>"),
                                self.period,
                                widgets.HTML("<b>Scale</b>"),
                                self.scale,
                                widgets.HTML("<b>Weight threshold</b>"),
                                self.threshold,
                            ]
                        ),
                        widgets.VBox(
                            [
                                widgets.HTML(f"<b>{topics_specifier} topics</b>"),
                                self.topic_ids,
                            ]
                        ),
                        widgets.VBox(
                            [
                                widgets.HTML("<b>Network layout</b>"),
                                self.layout_algorithm,
                                widgets.HTML("<b>Output</b>"),
                                self.output_format,
                                self.button,
                            ]
                        ),
                    ]
                ),
                self.output,
                self.text,
            ]
        )
        return _layout

    @dataclass
    class GUI_opts:
        plot_mode: PlotMode
        inferred_topics: InferredTopicsData
        layout_algorithm: str
        threshold: float
        period: Tuple[int, int]
        topic_ids: Sequence[int]
        scale: float
        output_format: str

    @property
    def opts(self) -> GUI_opts:
        return GUI.GUI_opts(
            plot_mode=self.plot_mode,
            inferred_topics=self.inferred_topics,
            layout_algorithm=self.layout_algorithm.value,
            threshold=self.threshold.value,
            period=self.period.value,
            topic_ids=self.topic_ids.value,
            scale=self.scale.value,
            output_format=self.output_format.value,
        )


def display_gui(plot_mode: PlotMode.FocusTopics, state: TopicModelContainer):

    gui: GUI = GUI(plot_mode=plot_mode).setup(inferred_topics=state.inferred_topics)
    display(gui.layout())
    # gui.update_handler()
