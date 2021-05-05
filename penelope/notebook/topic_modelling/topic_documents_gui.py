import warnings
from typing import Callable

import pandas as pd
import penelope.topic_modelling as topic_modelling
import penelope.utility as utility
from IPython.display import display
from ipywidgets import HTML, Button, FloatSlider, HBox, IntSlider, Label, Layout, Output, VBox
from penelope.corpus import DocumentIndex

from .model_container import TopicModelContainer
from .utility import filter_document_topic_weights

logger = utility.get_logger()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

TEXT_ID = "id_345"
BUTTON_STYLE = dict(description_width='initial', button_color='lightgreen')


class GUI:
    def __init__(self):

        self.n_topics: int = 0
        self.text_id: str = TEXT_ID
        self.text: HTML = HTML(value=f"<span class='{TEXT_ID}'></span>")
        self.topic_id: IntSlider = IntSlider(
            description='Topic ID', min=0, max=199, step=1, value=0, continuous_update=False
        )
        self.n_top: IntSlider = IntSlider(description='', min=5, max=500, step=1, value=75)
        self.threshold = FloatSlider(
            description='Threshold', min=0.0, max=1.0, step=0.01, value=0.20, continues_update=False
        )
        self.output: Output = Output()

        self.prev_topic_id: Button = Button(description="<<", layout=Layout(**BUTTON_STYLE))
        self.next_topic_id: Button = Button(description=">>", layout=Layout(**BUTTON_STYLE))

        self.callback: Callable = lambda *_: ()

    def _callback(self, *_):
        self.callback(self)

    def setup(self, *, n_topics, callback):

        self.callback = callback or self.callback
        self.n_topics = n_topics
        self.topic_id.value = 0
        self.topic_id.max = n_topics - 1

        self.prev_topic_id.on_click(self.goto_previous)
        self.next_topic_id.on_click(self.goto_next)

        self.topic_id.observe(self._callback, names='value')
        self.threshold.observe(self._callback, names='value')
        self.n_top.observe(self._callback, names='value')

        return self

    def goto_previous(self, *_):
        self.topic_id.value = (self.topic_id.value - 1) % self.topic_id.max

    def goto_next(self, *_):
        self.topic_id.value = (self.topic_id.value + 1) % self.topic_id.max

    def layout(self):
        return VBox(
            [
                HBox(
                    [
                        VBox(
                            [
                                HBox([self.prev_topic_id, self.next_topic_id]),
                                Label("Max number of documents to show"),
                                self.n_top,
                            ]
                        ),
                        VBox([self.topic_id, self.threshold]),
                    ]
                ),
                self.text,
                self.output,
            ]
        )


def get_topic_documents(
    document_topic_weights: pd.DataFrame,
    document_index: DocumentIndex,
    threshold: float = 0.0,
    n_top: int = 500,
    **filters,
) -> pd.DataFrame:
    topic_documents = filter_document_topic_weights(document_topic_weights, filters=filters, threshold=threshold)
    if len(topic_documents) == 0:
        return None

    topic_documents = (
        topic_documents.drop(['topic_id'], axis=1)
        .set_index('document_id')
        .sort_values('weight', ascending=False)
        .head(n_top)
    )
    additional_columns = [x for x in document_index.columns.tolist() if x not in ['year', 'document_name']]
    topic_documents = topic_documents.merge(
        document_index[additional_columns], left_index=True, right_on='document_id', how='inner'
    )
    topic_documents.index.name = 'id'
    return topic_documents


def get_topic_tokens(topic_token_weights: pd.DataFrame, topic_id: int):

    if len(topic_token_weights[topic_token_weights.topic_id == topic_id]) == 0:
        tokens = "Topics has no significant presence in any documents in the entire corpus"
    else:
        tokens = topic_modelling.get_topic_title(topic_token_weights, topic_id, n_tokens=200)

    return f'ID {topic_id}: {tokens}'


def display_gui(state: TopicModelContainer):

    topic_token_weights = state.inferred_topics.topic_token_weights
    document_topic_weights = state.inferred_topics.document_topic_weights
    document_index = state.inferred_topics.document_index

    def display_callback(gui: GUI):

        gui.output.clear_output()

        # if gui.n_topics != state.num_topics:
        #    gui.setup(n_topics)

        with gui.output:

            gui.text.value = get_topic_tokens(topic_token_weights, gui.topic_id.value)

            topic_documents = get_topic_documents(
                document_topic_weights=document_topic_weights,
                document_index=document_index,
                threshold=gui.threshold.value,
                n_top=gui.n_top.value,
                topic_id=gui.topic_id.value,
            )

            if topic_documents is not None:
                display(topic_documents)

    _gui = GUI().setup(n_topics=state.num_topics, callback=display_callback)

    display(_gui.layout())

    display_callback(_gui)
