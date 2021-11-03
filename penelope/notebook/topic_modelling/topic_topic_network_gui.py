from IPython.display import display
from ipywidgets import (  # type: ignore
    HTML,
    Dropdown,
    FloatSlider,
    HBox,
    IntProgress,
    IntRangeSlider,
    IntSlider,
    Output,
    SelectMultiple,
    VBox,
)

from .. import widgets_utils
from .model_container import TopicModelContainer
from .topic_topic_network_gui_utility import display_topic_topic_network

# bokeh.plotting.output_notebook()
TEXT_ID = 'nx_topic_topic'
LAYOUT_OPTIONS = ['Circular', 'Kamada-Kawai', 'Fruchterman-Reingold']
OUTPUT_OPTIONS = {'Network': 'network', 'Table': 'table', 'Excel': 'XLSX', 'CSV': 'CSV', 'Clipboard': 'clipboard'}

# pylint: disable=too-many-instance-attributes


class TopicTopicGUI:
    def __init__(self, state: TopicModelContainer):

        self.state: TopicModelContainer = state

        n_topics: int = self.state.num_topics

        ignore_options = [('', None)] + [('Topic #' + str(i), i) for i in range(0, n_topics)]
        year_min, year_max = state.inferred_topics.year_period

        self.n_topics = n_topics
        self.text = widgets_utils.text_widget(TEXT_ID)
        self.period: IntRangeSlider = IntRangeSlider(
            description='',
            min=year_min,
            max=year_max,
            step=1,
            value=(year_min, year_min + 5),
            continues_update=False,
        )
        self.scale: FloatSlider = FloatSlider(
            description='', min=0.0, max=1.0, step=0.01, value=0.1, continues_update=False
        )
        self.n_docs: IntSlider = IntSlider(description='', min=1, max=100, step=1, value=10, continues_update=False)
        self.threshold: FloatSlider = FloatSlider(
            description='', min=0.01, max=1.0, step=0.01, value=0.20, continues_update=False
        )
        self.output_format: Dropdown = Dropdown(
            description='', options=OUTPUT_OPTIONS, value='network', layout=dict(width='200px')
        )
        self.network_layout: Dropdown = Dropdown(
            description='', options=LAYOUT_OPTIONS, value='Fruchterman-Reingold', layout=dict(width='250px')
        )
        self.progress: IntProgress = IntProgress(min=0, max=4, step=1, value=0, layout=dict(width="99%"))
        self.ignores: SelectMultiple = SelectMultiple(
            description='', options=ignore_options, value=[], rows=10, layout=dict(width='250px')
        )
        self.node_range: IntRangeSlider = IntRangeSlider(
            description='', min=10, max=100, step=1, value=(20, 60), continues_update=False
        )
        self.edge_range: IntRangeSlider = IntRangeSlider(
            description='', min=1, max=20, step=1, value=(2, 6), continues_update=False
        )
        self.output: Output = Output()

        self.topic_proportions = self.state.inferred_topics.compute_topic_proportions()
        self.titles = self.state.inferred_topics.get_topic_titles()

    def layout(self) -> VBox:
        extra_widgets: VBox = self.extra_widgets()

        return VBox(
            [
                HBox(
                    [
                        VBox(
                            [
                                HTML("<b>Co-occurrence threshold</b>"),
                                self.threshold,
                                HTML("<b>Documents in common</b>"),
                                self.n_docs,
                                HTML("<b>Year range</b>"),
                                self.period,
                            ]
                        ),
                        VBox(
                            [
                                HTML("<b>Ignore topics</b>"),
                                self.ignores,
                            ]
                        ),
                    ]
                    + ([extra_widgets] if extra_widgets else [])
                    + [
                        VBox(
                            [
                                HTML("<b>Node size</b>"),
                                self.node_range,
                                HTML("<b>Edge size</b>"),
                                self.edge_range,
                                HTML("<b>Scale</b>"),
                                self.scale,
                            ]
                        ),
                        VBox(
                            [
                                HTML("<b>Network layout</b>"),
                                self.network_layout,
                                HTML("<b>Output</b>"),
                                self.output_format,
                                self.progress,
                            ]
                        ),
                    ]
                ),
                self.output,
                self.text,
            ]
        )

    def extra_widgets(self) -> VBox:
        return None

    def setup(self) -> "TopicTopicGUI":

        self.threshold.observe(self.update_handler, names='value')
        self.n_docs.observe(self.update_handler, names='value')
        self.period.observe(self.update_handler, names='value')
        self.scale.observe(self.update_handler, names='value')
        self.node_range.observe(self.update_handler, names='value')
        self.edge_range.observe(self.update_handler, names='value')
        self.output_format.observe(self.update_handler, names='value')
        self.network_layout.observe(self.update_handler, names='value')
        self.ignores.observe(self.update_handler, names='value')

        return self

    def update_handler(self, *_):

        self.output.clear_output()
        self.tick(1)
        with self.output:

            display_topic_topic_network(
                inferred_topics=self.state.inferred_topics,
                filters=self.get_data_filter(),
                period=self.period.value,
                ignores=self.ignores.value,
                threshold=self.threshold.value,
                layout=self.network_layout.value,
                n_docs=self.n_docs.value,
                scale=self.scale.value,
                node_range=self.node_range.value,
                edge_range=self.edge_range.value,
                output_format=self.output_format.value,
                element_id=TEXT_ID,
                titles=self.titles,
                topic_proportions=self.topic_proportions,
            )

        self.tick(0)

    def get_data_filter(self):
        return dict()

    def tick(self, x=None):
        self.progress.value = self.progress.value + 1 if x is None else x


def display_gui(state: TopicModelContainer):

    gui: TopicTopicGUI = TopicTopicGUI(state).setup()

    display(gui.layout())

    gui.update_handler()
