# Visualize topic co-occurrence
import types

import ipywidgets as widgets
import penelope.topic_modelling as topic_modelling
from IPython.display import display

from .. import widgets_utils
from .display_topic_topic_network import display_topic_topic_network
from .model_container import TopicModelContainer

# bokeh.plotting.output_notebook()


def display_gui(state: TopicModelContainer):

    lw = lambda w: widgets.Layout(width=w)
    n_topics = state.num_topics

    text_id = 'nx_topic_topic'
    layout_options = ['Circular', 'Kamada-Kawai', 'Fruchterman-Reingold']
    output_options = {'Network': 'network', 'Table': 'table', 'Excel': 'excel', 'CSV': 'csv'}
    ignore_options = [('', None)] + [('Topic #' + str(i), i) for i in range(0, n_topics)]
    year_min, year_max = state.inferred_topics.year_period

    topic_proportions = state.inferred_topics.compute_topic_proportions()

    titles = topic_modelling.get_topic_titles(state.inferred_topics.topic_token_weights)

    gui = types.SimpleNamespace(
        n_topics=n_topics,
        text=widgets_utils.text_widget(text_id),
        period=widgets.IntRangeSlider(
            description='',
            min=year_min,
            max=year_max,
            step=1,
            value=(year_min, year_min + 5),
            continues_update=False,
        ),
        scale=widgets.FloatSlider(description='', min=0.0, max=1.0, step=0.01, value=0.1, continues_update=False),
        n_docs=widgets.IntSlider(description='', min=1, max=100, step=1, value=10, continues_update=False),
        threshold=widgets.FloatSlider(description='', min=0.01, max=1.0, step=0.01, value=0.20, continues_update=False),
        output_format=widgets.Dropdown(description='', options=output_options, value='network', layout=lw('200px')),
        layout=widgets.Dropdown(
            description='', options=layout_options, value='Fruchterman-Reingold', layout=lw('250px')
        ),
        progress=widgets.IntProgress(min=0, max=4, step=1, value=0, layout=widgets.Layout(width="99%")),
        ignores=widgets.SelectMultiple(description='', options=ignore_options, value=[], rows=10, layout=lw('250px')),
        node_range=widgets.IntRangeSlider(
            description='', min=10, max=100, step=1, value=(20, 60), continues_update=False
        ),
        edge_range=widgets.IntRangeSlider(description='', min=1, max=20, step=1, value=(2, 6), continues_update=False),
        output=widgets.Output(),
    )

    def tick(x=None):
        gui.progress.value = gui.progress.value + 1 if x is None else x

    def compute_handler(*_):

        gui.output.clear_output()
        tick(1)
        with gui.output:

            display_topic_topic_network(
                inferred_topics=state.inferred_topics,
                filters=dict(),
                period=gui.period.value,
                ignores=gui.ignores.value,
                threshold=gui.threshold.value,
                layout=gui.layout.value,
                n_docs=gui.n_docs.value,
                scale=gui.scale.value,
                node_range=gui.node_range.value,
                edge_range=gui.edge_range.value,
                output_format=gui.output_format.value,
                text_id=text_id,
                titles=titles,
                topic_proportions=topic_proportions,
            )
        tick(0)

    gui.threshold.observe(compute_handler, names='value')
    gui.n_docs.observe(compute_handler, names='value')
    gui.period.observe(compute_handler, names='value')
    gui.scale.observe(compute_handler, names='value')
    gui.node_range.observe(compute_handler, names='value')
    gui.edge_range.observe(compute_handler, names='value')
    gui.output_format.observe(compute_handler, names='value')
    gui.layout.observe(compute_handler, names='value')
    gui.ignores.observe(compute_handler, names='value')

    display(
        widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                widgets.HTML("<b>Co-occurrence threshold</b>"),
                                gui.threshold,
                                widgets.HTML("<b>Documents in common</b>"),
                                gui.n_docs,
                                widgets.HTML("<b>Year range</b>"),
                                gui.period,
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
                                widgets.HTML("<b>Node size</b>"),
                                gui.node_range,
                                widgets.HTML("<b>Edge size</b>"),
                                gui.edge_range,
                                widgets.HTML("<b>Scale</b>"),
                                gui.scale,
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
    )

    compute_handler()
