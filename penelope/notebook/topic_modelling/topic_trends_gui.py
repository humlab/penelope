import ipywidgets as w  # type: ignore
import pandas as pd
from IPython.display import display

import penelope.topic_modelling as tm

from . import mixins as mx
from . import topic_trends_gui_utility as gui_utils
from .model_container import TopicModelContainer


class TopicTrendsGUI(mx.NextPrevTopicMixIn, mx.TopicsStateGui):
    def __init__(self, state: TopicModelContainer, calculator: tm.TopicPrevalenceOverTimeCalculator = None):
        super().__init__(state=state)

        timespan: tuple = self.inferred_topics.timespan

        self._text = w.HTML(value="", placeholder='', description='')

        self._aggregate: w.Dropdown = w.Dropdown(
            description='Aggregate',
            options=[(x['description'], x['key']) for x in tm.YEARLY_AVERAGE_COMPUTE_METHODS],
            value='true_average_weight',
            layout=dict(width="200px"),
        )

        self._normalize: w.ToggleButton = w.ToggleButton(
            description='Normalize', value=True, layout=dict(width="120px")
        )
        self._year_range: w.IntRangeSlider = w.IntRangeSlider(min=timespan[0], max=timespan[1], value=timespan)
        self._output_format: w.Dropdown = w.Dropdown(
            description='Format', options=['Chart', 'Table'], value='Chart', layout=dict(width="200px")
        )

        self._output: w.Output = w.Output()
        self._extra_placeholder: w.VBox = w.HBox()

        self.calculator: tm.TopicPrevalenceOverTimeCalculator = calculator or tm.prevelance.default_calculator()

    def layout(self):
        return w.VBox(
            [
                w.HBox(
                    [
                        w.VBox(
                            [
                                w.HBox([self._prev_topic_id, self._next_topic_id]),
                                self._year_range,
                            ]
                        ),
                        w.VBox([self.topic_id]),
                        self._extra_placeholder,
                        w.VBox([self._aggregate, self._output_format]),
                        w.VBox([self._normalize]),
                    ]
                ),
                self._text,
                self._output,
            ]
        )

    def setup(self, **kwargs) -> "TopicTrendsGUI":
        super().setup(**kwargs)

        self.topic_id = (0, self.inferred_n_topics - 1)

        self._topic_id.observe(self.update_handler, names='value')
        self._normalize.observe(self.update_handler, names='value')
        self._aggregate.observe(self.update_handler, names='value')
        self._output_format.observe(self.update_handler, names='value')

        return self

    def topic_changed(self, topic_id: int):

        tokens = self.inferred_topics.get_topic_title(topic_id, n_tokens=200)

        self._text.value = 'ID {}: {}'.format(topic_id, tokens)

    def compute_weights(self) -> pd.DataFrame:
        return self.calculator.compute(
            inferred_topics=self.inferred_topics,
            filters=self.get_filters(),
            threshold=self.get_threshold(),
            result_threshold=self.get_result_threshold(),
        )

    def get_filters(self) -> dict:
        return {'year': self.year_range}

    def get_threshold(self) -> float:
        return 0.0

    def get_result_threshold(self) -> float:
        return 0.0

    @property
    def year_range(self) -> tuple:
        return self._year_range.value

    @property
    def aggregate(self) -> tuple:
        return self._aggregate.value

    @property
    def normalize(self) -> tuple:
        return self._normalize.value

    @property
    def output_format(self) -> tuple:
        return self._output_format.value

    def update_handler(self, *_):

        self._output.clear_output()

        with self._output:

            self.topic_changed(self.topic_id)

            weights = self.compute_weights()

            gui_utils.display_topic_trends(
                weight_over_time=weights,
                topic_id=self.topic_id,
                year_range=self.year_range,
                aggregate=self.aggregate,
                normalize=self.normalize,
                output_format=self.output_format,
            )


def display_gui(state: TopicModelContainer):

    gui = TopicTrendsGUI(state=state, calculator=None).setup()

    display(gui.layout())

    gui.update_handler()

    return gui
