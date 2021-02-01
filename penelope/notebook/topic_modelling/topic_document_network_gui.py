# Visualize year-to-topic network by means of topic-document-weights
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Sequence, Tuple

import bokeh
import ipywidgets as widgets
import networkx as nx
import pandas as pd
import penelope.network.metrics as network_metrics
import penelope.network.plot_utility as network_plot
import penelope.notebook.widgets_utils as widget_utils
from bokeh.models.sources import ColumnDataSource
from IPython import display
from penelope import topic_modelling, utility
from penelope.network import layout_source
from penelope.network.networkx import utility as network_utility
from penelope.topic_modelling.container import InferredTopicsData

from .display_utility import display_document_topics_as_grid
from .model_container import TopicModelContainer

logger = utility.getLogger("westac")

NETWORK_LAYOUT_ALGORITHMS = ["Circular", "Kamada-Kawai", "Fruchterman-Reingold"]


class PlotMode(IntEnum):
    Default = 1
    FocusTopics = 2


def plot_document_topic_network(
    network: nx.Graph, layout_data, scale: float = 1.0, titles=None, highlight_topic_ids=None, text_id: str = "nx_id1"
):  # pylint: disable=unused-argument, too-many-locals

    tools: str = "pan,wheel_zoom,box_zoom,reset,hover,save"

    document_nodes, topic_nodes = network_utility.get_bipartite_node_set(network, bipartite=0)

    color_map: dict = (
        {x: "brown" if x in highlight_topic_ids else "skyblue" for x in topic_nodes}
        if highlight_topic_ids is not None
        else None
    )
    color_specifier: str = "colors" if highlight_topic_ids is not None else "skyblue"

    document_source: ColumnDataSource = layout_source.create_nodes_subset_data_source(
        network, layout_data, document_nodes
    )
    topic_source: ColumnDataSource = layout_source.create_nodes_subset_data_source(
        network, layout_data, topic_nodes, color_map=color_map
    )
    lines_source: ColumnDataSource = layout_source.create_edges_layout_data_source(
        network, layout_data, scale=6.0, normalize=False
    )

    edges_alphas: List[float] = network_metrics.compute_alpha_vector(lines_source.data["weights"])

    lines_source.add(edges_alphas, "alphas")

    p = bokeh.plotting.figure(plot_width=1000, plot_height=600, x_axis_type=None, y_axis_type=None, tools=tools)

    _ = p.multi_line(
        xs="xs", ys="ys", line_width="weights", alpha="alphas", level="underlay", color="black", source=lines_source
    )
    _ = p.circle(x="x", y="y", size=40, source=document_source, color="lightgreen", line_width=1, alpha=1.0)

    r_topics = p.circle(x="x", y="y", size=25, source=topic_source, color=color_specifier, alpha=1.00)

    callback = widget_utils.glyph_hover_callback2(
        topic_source, "node_id", text_ids=titles.index, text=titles, element_id=text_id
    )

    p.add_tools(bokeh.models.HoverTool(renderers=[r_topics], tooltips=None, callback=callback))

    text_opts = dict(x="x", y="y", text="name", level="overlay", x_offset=0, y_offset=0, text_font_size="8pt")

    p.add_layout(
        bokeh.models.LabelSet(
            source=document_source, text_color="black", text_align="center", text_baseline="middle", **text_opts
        )
    )
    p.add_layout(
        bokeh.models.LabelSet(
            source=topic_source, text_color="black", text_align="center", text_baseline="middle", **text_opts
        )
    )

    return p


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

        titles = topic_modelling.get_topic_titles(opts.inferred_topics.topic_token_weights)

        p = plot_document_topic_network(
            network,
            layout_data,
            scale=opts.scale,
            titles=titles,
            highlight_topic_ids=None if opts.plot_mode is None else opts.topic_ids,
            text_id=f"ID_{opts.plot_mode.name}",
        )

        bokeh.plotting.show(p)

    elif opts.output_format == "table":
        g = display_document_topics_as_grid(df_network)
        display.display(g)


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


@dataclass
class GUI:
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

    plot_mode: PlotMode
    inferred_topics: InferredTopicsData = None

    text: widgets.HTML = None
    period = widgets.IntRangeSlider(
        description="",
        min=1900,
        max=2030,
        step=1,
        value=(1900, 1900 + 5),
        continues_update=False,
    )
    scale = widgets.FloatSlider(
        description="",
        min=0.0,
        max=1.0,
        step=0.01,
        value=0.1,
        continues_update=False,
    )
    threshold = widgets.FloatSlider(
        description="",
        min=0.0,
        max=1.0,
        step=0.01,
        value=0.10,
        continues_update=False,
    )
    output_format = widgets.Dropdown(
        description="",
        options={"Network": "network", "Table": "table"},
        value="network",
        layout={'width': '200px'},
    )
    layout_algorithm = widgets.Dropdown(description="", options=[], value=None, layout={'width': '250px'})
    topic_ids = widgets.SelectMultiple(description="", options=[], value=[], rows=8, layout={'width': '180px'})
    button = widgets.Button(
        description="Display", button_style='Success', layout=widgets.Layout(width='115px', background_color='blue')
    )
    output = widgets.Output()

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

        self.text = widget_utils.text_widget(f"ID_{self.plot_mode.name}")
        self.button.on_click(self.update_handler)
        # self.threshold.observe(self.update_handler, names='value')
        # self.period.observe(self.update_handler, names='value')
        # self.scale.observe(self.update_handler, names='value')
        # self.output_format.observe(self.update_handler, names='value')
        # self.layout_algorithm.observe(self.update_handler, names='value')
        # self.topic_ids.observe(self.update_handler, names='value')
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


def display_gui(plot_mode: PlotMode.FocusTopics, state: TopicModelContainer):

    gui: GUI = GUI(plot_mode=plot_mode).setup(inferred_topics=state.inferred_topics)
    display.display(gui.layout())
    # gui.update_handler()
