# Visualize year-to-topic network by means of topic-document-weights
import types

import bokeh
import ipywidgets as widgets
import penelope.network.metrics as network_metrics
import penelope.network.plot_utility as network_plot
import penelope.network.utility as network_utility
import penelope.notebook.widgets_utils as widget_utils
import penelope.topic_modelling as topic_modelling
import penelope.utility as utility
from IPython.display import display

from .display_utility import display_document_topics_as_grid
from .model_container import TopicModelContainer

logger = utility.getLogger("westac")

# FIXME: Consolidate document topic network files


def plot_document_topic_network(
    network, layout, scale=1.0, titles=None
):  # pylint: disable=unused-argument, too-many-locals
    tools = "pan,wheel_zoom,box_zoom,reset,hover,save"
    year_nodes, topic_nodes = network_utility.get_bipartite_node_set(network, bipartite=0)

    year_source = network_utility.get_node_subset_source(network, layout, year_nodes)
    topic_source = network_utility.get_node_subset_source(network, layout, topic_nodes)
    lines_source = network_utility.get_edges_source(network, layout, scale=6.0, normalize=False)

    edges_alphas = network_metrics.compute_alpha_vector(lines_source.data["weights"])

    lines_source.add(edges_alphas, "alphas")

    p = bokeh.plotting.figure(
        plot_width=1000,
        plot_height=600,
        x_axis_type=None,
        y_axis_type=None,
        tools=tools,
    )

    _ = p.multi_line(
        xs="xs",
        ys="ys",
        line_width="weights",
        alpha="alphas",
        level="underlay",
        color="black",
        source=lines_source,
    )
    _ = p.circle(
        x="x",
        y="y",
        size=40,
        source=year_source,
        color="lightgreen",
        line_width=1,
        alpha=1.0,
    )

    r_topics = p.circle(x="x", y="y", size=25, source=topic_source, color="skyblue", alpha=1.00)

    callback = widget_utils.glyph_hover_callback2(
        topic_source, "node_id", text_ids=titles.index, text=titles, element_id="nx_id1"
    )

    p.add_tools(bokeh.models.HoverTool(renderers=[r_topics], tooltips=None, callback=callback))

    text_opts = dict(
        x="x",
        y="y",
        text="name",
        level="overlay",
        x_offset=0,
        y_offset=0,
        text_font_size="8pt",
    )

    p.add_layout(
        bokeh.models.LabelSet(
            source=year_source, text_color="black", text_align="center", text_baseline="middle", **text_opts
        )
    )
    p.add_layout(
        bokeh.models.LabelSet(
            source=topic_source, text_color="black", text_align="center", text_baseline="middle", **text_opts
        )
    )

    return p


def display_document_topic_network(  # pylint: disable=too-many-locals)
    layout_algorithm,
    inferred_topics: topic_modelling.InferredTopicsData,
    threshold=0.10,
    period=None,
    ignores=None,
    scale=1.0,
    output_format="network",
    tick=utility.noop,
):

    tick(1)

    topic_token_weights = inferred_topics.topic_token_weights
    document_topic_weights = inferred_topics.document_topic_weights

    titles = topic_modelling.get_topic_titles(topic_token_weights)

    df = document_topic_weights[document_topic_weights.weight > threshold].reset_index()

    if len(period or []) == 2:
        df = df[(df.year >= period[0]) & (df.year <= period[1])]

    if len(ignores or []) > 0:
        df = df[~df.topic_id.isin(ignores)]

    df["weight"] = utility.clamp_values(list(df.weight), (0.1, 2.0))

    if len(df) == 0:
        print("No data")
        return

    if "filename" not in df:
        df = df.merge(
            inferred_topics.document_index["filename"],
            left_on="document_id",
            right_on="document_id",
        )

    df["title"] = df.filename

    network = network_utility.create_bipartite_network(df, "title", "topic_id")
    tick()

    if output_format == "network":
        if layout_algorithm == "Circular":
            args = dict(dim=2, center=None, scale=1.0)
        else:
            args = network_plot.layout_args(layout_algorithm, network, scale)
        layout = (network_plot.layout_algorithms[layout_algorithm])(network, **args)
        tick()
        p = plot_document_topic_network(network, layout, scale=scale, titles=titles)
        bokeh.plotting.show(p)

    elif output_format == "table":
        g = display_document_topics_as_grid(df)
        display(g)

    tick(0)


def display_gui(state: TopicModelContainer):

    lw = lambda w: widgets.Layout(width=w)

    inferred_topics: topic_modelling.InferredTopicsData = state.inferred_topics

    text_id = "nx_id1"
    layout_options = ["Circular", "Kamada-Kawai", "Fruchterman-Reingold"]
    year_min, year_max = inferred_topics.year_period

    n_topics = state.num_topics

    gui = types.SimpleNamespace(
        text=widget_utils.text_widget(text_id),
        period=widgets.IntRangeSlider(
            description="",
            min=year_min,
            max=year_max,
            step=1,
            value=(year_min, year_min + 5),
            continues_update=False,
        ),
        scale=widgets.FloatSlider(
            description="",
            min=0.0,
            max=1.0,
            step=0.01,
            value=0.1,
            continues_update=False,
        ),
        threshold=widgets.FloatSlider(
            description="",
            min=0.0,
            max=1.0,
            step=0.01,
            value=0.50,
            continues_update=False,
        ),
        output_format=widgets.Dropdown(
            description="",
            options={"Network": "network", "Table": "table"},
            value="network",
            layout=lw("200px"),
        ),
        layout=widgets.Dropdown(
            description="",
            options=layout_options,
            value="Fruchterman-Reingold",
            layout=lw("250px"),
        ),
        progress=widgets.IntProgress(min=0, max=4, step=1, value=0, layout=widgets.Layout(width="99%")),
        ignores=widgets.SelectMultiple(
            description="",
            options=[("", None)] + [("Topic #" + str(i), i) for i in range(0, n_topics)],
            value=[],
            rows=8,
            layout=lw("180px"),
        ),
        output=widgets.Output(),
    )

    def tick(x=None):
        gui.progress.value = gui.progress.value + 1 if x is None else x

    def update_handler(*_):

        gui.output.clear_output()

        with gui.output:

            display_document_topic_network(
                layout_algorithm=gui.layout.value,
                inferred_topics=inferred_topics,
                threshold=gui.threshold.value,
                period=gui.period.value,
                ignores=gui.ignores.value,
                scale=gui.scale.value,
                output_format=gui.output_format.value,
                tick=tick,
            )

    gui.threshold.observe(update_handler, names='value')
    gui.period.observe(update_handler, names='value')
    gui.scale.observe(update_handler, names='value')
    gui.output_format.observe(update_handler, names='value')
    gui.layout.observe(update_handler, names='value')
    gui.ignores.observe(update_handler, names='value')

    # FIXME: Add button

    _layout = widgets.VBox(
        [
            widgets.HBox(
                [
                    widgets.VBox(
                        [
                            widgets.HTML("<b>Year range</b>"),
                            gui.period,
                            widgets.HTML("<b>Scale</b>"),
                            gui.scale,
                            widgets.HTML("<b>Weight threshold</b>"),
                            gui.threshold,
                        ]
                    ),
                    widgets.VBox(
                        [
                            widgets.HTML("<b>Ignore topics</b>"),
                            gui.ignores,
                        ]
                    ),
                    widgets.VBox(
                        [
                            widgets.HTML("<b>Network layout</b>"),
                            gui.layout,
                            widgets.HTML("<b>Output</b>"),
                            gui.output_format,
                            gui.progress,
                        ]
                    ),
                ]
            ),
            gui.output,
            gui.text,
        ]
    )

    display(_layout)
    update_handler()
