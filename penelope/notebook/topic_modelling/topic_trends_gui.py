import warnings
from typing import Optional

import penelope.topic_modelling as topic_modelling
from IPython.display import display
from ipywidgets import Button, Dropdown, HBox, IntProgress, IntSlider, Output, ToggleButton, VBox

from .. import widgets_utils
from .display_topic_trends import display_topic_trends
from .model_container import TopicModelContainer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

TEXT_ID = 'topic_share_plot'


class GUI:  # pylint: disable=too-many-instance-attributes
    def __init__(self):

        self.state: TopicModelContainer = None

        self.text = widgets_utils.text_widget(TEXT_ID)
        self.n_topics: Optional[int] = None
        self.text_id = TEXT_ID

        self.aggregate = Dropdown(
            description='Aggregate',
            options=[(x['description'], x['key']) for x in topic_modelling.YEARLY_MEAN_COMPUTE_METHODS],
            value='true_mean',
            layout=dict(width="200px"),
        )

        self.normalize = ToggleButton(description='Normalize', value=True, layout=dict(width="120px"))
        self.topic_id = IntSlider(description='Topic ID', min=0, max=999, step=1, value=0, continuous_update=False)

        self.output_format = Dropdown(
            description='Format', options=['Chart', 'Table'], value='Chart', layout=dict(width="200px")
        )

        self.progress = IntProgress(min=0, max=4, step=1, value=0)
        self.output = Output()

        self.prev_topic_id: Optional[Button] = None
        self.next_topic_id: Optional[Button] = None

    def layout(self):
        return VBox(
            children=[
                HBox(
                    children=[
                        VBox(
                            children=[
                                HBox(children=[self.prev_topic_id, self.next_topic_id]),
                                self.progress,
                            ]
                        ),
                        VBox(children=[self.topic_id]),
                        VBox(children=[self.aggregate, self.output_format]),
                        VBox(children=[self.normalize]),
                    ]
                ),
                self.text,
                self.output,
            ]
        )

    def setup(self, state: TopicModelContainer) -> "GUI":

        self.state = state
        self.topic_id.max = state.num_topics - 1
        self.prev_topic_id = widgets_utils.button_with_previous_callback(self, 'topic_id', state.num_topics)
        self.next_topic_id = widgets_utils.button_with_next_callback(self, 'topic_id', state.num_topics)
        self.topic_id.observe(self.update_handler, names='value')
        self.normalize.observe(self.update_handler, names='value')
        self.aggregate.observe(self.update_handler, names='value')
        self.output_format.observe(self.update_handler, names='value')

        return self

    def on_topic_change_update_gui(self, topic_id: int):

        if self.n_topics != self.state.num_topics:
            self.n_topics = self.state.num_topics
            self.topic_id.value = 0
            self.topic_id.max = self.state.num_topics - 1

        tokens = topic_modelling.get_topic_title(self.state.inferred_topics.topic_token_weights, topic_id, n_tokens=200)

        self.text.value = 'ID {}: {}'.format(topic_id, tokens)

    def update_handler(self, *_):

        self.output.clear_output()

        with self.output:

            self.on_topic_change_update_gui(self.topic_id.value)

            weights = topic_modelling.compute_topic_yearly_means(self.state.inferred_topics.document_topic_weights)

            display_topic_trends(
                weight_over_time=weights,
                topic_id=self.topic_id.value,
                year_range=self.state.inferred_topics.year_period,
                aggregate=self.aggregate.value,
                normalize=self.normalize.value,
                output_format=self.output_format.value,
            )


def display_gui(state: TopicModelContainer):

    gui = GUI().setup(state)

    display(gui.layout())

    gui.update_handler()
