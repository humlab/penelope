from typing import Optional

import pandas as pd
import penelope.topic_modelling as tm
from IPython.display import display
from ipywidgets import Button, Dropdown, HBox, IntProgress, IntSlider, Output, ToggleButton, VBox  # type: ignore

from .. import widgets_utils
from . import topic_trends_gui_utility as gui_utils
from .model_container import TopicModelContainer

TEXT_ID = 'topic_share_plot'


class TopicTrendsGUI:
    def __init__(self, calculator: tm.TopicPrevalenceOverTimeCalculator = None):
        super().__init__()

        self.state: TopicModelContainer = None

        self.text = widgets_utils.text_widget(TEXT_ID)
        self.n_topics: Optional[int] = None
        self.text_id = TEXT_ID

        self.aggregate = Dropdown(
            description='Aggregate',
            options=[(x['description'], x['key']) for x in tm.YEARLY_AVERAGE_COMPUTE_METHODS],
            value='true_average_weight',
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

        self.extra_placeholder: VBox = HBox()

        self.calculator: tm.TopicPrevalenceOverTimeCalculator = calculator or tm.prevelance.default_calculator()

    def layout(self):
        return VBox(
            [
                HBox(
                    [
                        VBox(
                            [
                                HBox([self.prev_topic_id, self.next_topic_id]),
                                self.progress,
                            ]
                        ),
                        VBox([self.topic_id]),
                        self.extra_placeholder,
                        VBox([self.aggregate, self.output_format]),
                        VBox([self.normalize]),
                    ]
                ),
                self.text,
                self.output,
            ]
        )

    def setup(self, state: TopicModelContainer) -> "TopicTrendsGUI":

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

        tokens = self.state.inferred_topics.get_topic_title(topic_id, n_tokens=200)

        self.text.value = 'ID {}: {}'.format(topic_id, tokens)

    def compute_weights(self) -> pd.DataFrame:
        return self.calculator.compute(
            inferred_topics=self.state.inferred_topics,
            filters=self.get_filters(),
            threshold=self.get_threshold(),
            result_threshold=self.get_result_threshold(),
        )

    def get_filters(self) -> dict:
        return {}

    def get_threshold(self) -> float:
        return 0.0

    def get_result_threshold(self) -> float:
        return 0.0

    def update_handler(self, *_):

        self.output.clear_output()

        with self.output:

            self.on_topic_change_update_gui(self.topic_id.value)

            weights = self.compute_weights()

            gui_utils.display_topic_trends(
                weight_over_time=weights,
                topic_id=self.topic_id.value,
                year_range=self.state.inferred_topics.year_period,
                aggregate=self.aggregate.value,
                normalize=self.normalize.value,
                output_format=self.output_format.value,
            )


def display_gui(state: TopicModelContainer):

    gui = TopicTrendsGUI(calculator=None).setup(state)

    display(gui.layout())

    gui.update_handler()

    return gui
