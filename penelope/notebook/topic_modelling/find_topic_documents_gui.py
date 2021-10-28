from typing import Callable

import pandas as pd
from IPython.display import display
from ipywidgets import HTML, FloatSlider, HBox, IntSlider, Output, Text, VBox  # type: ignore
from penelope.topic_modelling import InferredTopicsData  # type: ignore

from .display_topic_titles import reduce_topic_tokens_overview
from .model_container import TopicModelContainer


class FindTopicDocumentsGUI:
    def __init__(self, state: TopicModelContainer):
        self.state: TopicModelContainer = state
        self.threshold_slider: FloatSlider = FloatSlider(min=0.01, max=1.0, value=0.2)
        self.top_token_slider: IntSlider = IntSlider(min=3, max=200, value=3, disabled=True)
        self.find_text: Text = Text(description="")
        self.output: Output = Output()
        self.callback = lambda *_: ()
        self.toplist_label: HTML = HTML("Tokens toplist threshold for token")

    def layout(self):
        return VBox(
            (
                HBox(
                    (
                        VBox(
                            (
                                HTML("Topic weight in document threshold"),
                                self.threshold_slider,
                            )
                        ),
                        VBox(
                            (
                                HTML("<b>Find topics containing token</b>"),
                                self.find_text,
                            )
                        ),
                        VBox(
                            (
                                self.toplist_label,
                                self.top_token_slider,
                            )
                        ),
                    )
                ),
                self.output,
            )
        )

    def _find_text(self, *_):
        self.top_token_slider.disabled = len(self.find_text.value) < 2

    def setup(self, callback: Callable):
        self.threshold_slider.observe(self.update_handler, 'value')
        self.top_token_slider.observe(self.update_handler, 'value')
        self.find_text.observe(self.update_handler, 'value')
        self.find_text.observe(self._find_text, 'value')
        self.callback = callback or self.callback
        return self

    @property
    def threshold(self) -> float:
        return self.threshold_slider.value

    @property
    def text(self) -> str:
        return self.find_text.value

    @property
    def n_top(self) -> int:
        return self.top_token_slider.value

    def update_handler(self, *_):

        inferred_topics: InferredTopicsData = self.state.inferred_topics

        self.toplist_label.value = f"<b>Token must be within top {self.top_token_slider.value} topic tokens</b>"
        self.output.clear_output()

        with self.output:

            document_topics: pd.DataFrame = self.callback(
                document_topic_weights=inferred_topics.document_topic_weights,
                topic_token_overview=inferred_topics.topic_token_overview,
                text=self.text,
                n_top=self.n_top,
                threshold=self.threshold,
            )

            display(document_topics)


def get_document_topic_weights(
    document_topic_weights: pd.DataFrame,
    topic_token_overview: pd.DataFrame,
    text: str,
    n_top: int,
    threshold: float,
) -> pd.DataFrame:

    weights: pd.DataFrame = document_topic_weights

    if len(text) > 2:

        topic_ids = reduce_topic_tokens_overview(topic_token_overview, n_top, text).index.tolist()

        weights = weights[weights.topic_id.isin(topic_ids)]

    weights = weights[weights.weight >= threshold]

    return weights


def create_gui(state: TopicModelContainer) -> FindTopicDocumentsGUI:

    gui: FindTopicDocumentsGUI = FindTopicDocumentsGUI(state).setup(callback=get_document_topic_weights)

    display(gui.layout())

    return gui
