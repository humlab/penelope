import pandas as pd
import penelope.topic_modelling as tm
from IPython.display import display
from ipywidgets import HTML, Dropdown, HBox, Output, ToggleButton, VBox  # type: ignore

from .. import widgets_utils
from .display_topic_trends_heatmap import display_heatmap
from .model_container import TopicModelContainer

TEXT_ID = 'topic_relevance'


class TopicOverviewGUI:
    def __init__(self):

        weighings = [(x['description'], x['key']) for x in tm.YEARLY_MEAN_COMPUTE_METHODS]

        self.state: TopicModelContainer = None
        self.text_id: str = TEXT_ID
        self.text: HTML = widgets_utils.text_widget(TEXT_ID)
        self.flip_axis: ToggleButton = ToggleButton(value=False, description='Flip', icon='', layout=dict(width="80px"))
        self.aggregate: Dropdown = Dropdown(
            description='Aggregate', options=weighings, value='max_weight', layout=dict(width="250px")
        )
        self.output_format: Dropdown = Dropdown(
            description='Output', options=['Heatmap', 'Table'], value='Heatmap', layout=dict(width="180px")
        )
        self.output: Output = Output()
        self.titles: pd.DataFrame = None

    def setup(self, state: TopicModelContainer) -> "TopicOverviewGUI":
        self.state = state
        self.aggregate.observe(self.update_handler, names='value')
        self.output_format.observe(self.update_handler, names='value')
        self.flip_axis.observe(self.update_handler, names='value')
        self.titles: pd.DataFrame = tm.get_topic_titles(self.state.inferred_topics.topic_token_weights, n_tokens=100)
        return self

    def update_handler(self, *_):

        self.output.clear_output()
        self.flip_axis.disabled = True
        self.flip_axis.description = 'Wait!'

        with self.output:

            weights: pd.DataFrame = self.compute_weights()

            display_heatmap(
                weights,
                self.titles,
                flip_axis=self.flip_axis.value,
                aggregate=self.aggregate.value,
                output_format=self.output_format.value,
            )

        self.flip_axis.disabled = False
        self.flip_axis.description = 'Flip'

    def layout(self) -> VBox:
        return VBox([HBox([self.aggregate, self.output_format, self.flip_axis]), HBox([self.output]), self.text])

    def compute_weights(self) -> pd.DataFrame:
        weights = tm.compute_topic_yearly_means(self.state.inferred_topics.document_topic_weights).fillna(0)
        return weights


def display_gui(state: TopicModelContainer):

    gui: TopicOverviewGUI = TopicOverviewGUI().setup(state)
    display(gui.layout())
    gui.update_handler()
