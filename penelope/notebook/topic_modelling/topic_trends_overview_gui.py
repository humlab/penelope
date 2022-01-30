import pandas as pd
from IPython.display import display
from ipywidgets import HTML, Dropdown, HBox, Output, ToggleButton, VBox  # type: ignore

from penelope import topic_modelling as tm

from .. import widgets_utils
from . import mixins as mx
from .model_container import TopicModelContainer
from .topic_trends_overview_gui_utility import display_heatmap

TEXT_ID = 'topic_relevance'


class TopicOverviewGUI(mx.TopicsStateGui):
    def __init__(self, state: TopicModelContainer, calculator: tm.MemoizedTopicPrevalenceOverTimeCalculator):
        super().__init__(state=state)

        self.titles: pd.DataFrame = None
        self.calculator: tm.MemoizedTopicPrevalenceOverTimeCalculator = calculator

        weighings = [(x['description'], x['key']) for x in tm.YEARLY_AVERAGE_COMPUTE_METHODS]

        self._text_id: str = TEXT_ID
        self._text: HTML = widgets_utils.text_widget(TEXT_ID)
        self._flip_axis: ToggleButton = ToggleButton(
            value=False, description='Flip', icon='', layout=dict(width="80px")
        )
        self._aggregate: Dropdown = Dropdown(
            description='Aggregate', options=weighings, value='max_weight', layout=dict(width="250px")
        )
        self._output_format: Dropdown = Dropdown(
            description='Output', options=['Heatmap', 'Table'], value='Heatmap', layout=dict(width="180px")
        )
        self._output: Output = Output()

    def setup(self) -> "TopicOverviewGUI":
        self._aggregate.observe(self.update_handler, names='value')
        self._output_format.observe(self.update_handler, names='value')
        self._flip_axis.observe(self.update_handler, names='value')
        self.titles: pd.DataFrame = self.inferred_topics.get_topic_titles(n_tokens=100)
        return self

    def update_handler(self, *_):

        self._output.clear_output()
        self._flip_axis.disabled = True
        self._flip_axis.description = 'Wait!'

        with self._output:

            weights: pd.DataFrame = self.compute_weights()

            display_heatmap(
                weights,
                self.titles,
                flip_axis=self._flip_axis.value,
                aggregate=self._aggregate.value,
                output_format=self._output_format.value,
            )

        self._flip_axis.disabled = False
        self._flip_axis.description = 'Flip'

    def layout(self) -> VBox:
        return VBox([HBox([self._aggregate, self._output_format, self._flip_axis]), HBox([self._output]), self._text])

    def compute_weights(self) -> pd.DataFrame:
        return self.calculator.compute(
            inferred_topics=self.inferred_topics,
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


def display_gui(state: TopicModelContainer):
    calculator: tm.MemoizedTopicPrevalenceOverTimeCalculator = tm.MemoizedTopicPrevalenceOverTimeCalculator(
        calculator=tm.AverageTopicPrevalenceOverTimeCalculator()
    )
    gui: TopicOverviewGUI = TopicOverviewGUI(state=state, calculator=calculator).setup()
    display(gui.layout())
    gui.update_handler()
