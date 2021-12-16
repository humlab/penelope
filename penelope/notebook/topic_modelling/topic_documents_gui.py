import warnings

import penelope.topic_modelling as tm
from IPython.display import display
from ipywidgets import HTML, Button, FloatSlider, HBox, IntSlider, Label, Layout, Output, VBox  # type: ignore

from .model_container import TopicModelContainer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

TEXT_ID = "id_345"
BUTTON_STYLE = dict(description_width='initial', button_color='lightgreen')


class TopicDocumentsGUI:
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

        self.state: TopicModelContainer = None

    def setup(self, *, state: TopicModelContainer) -> "TopicDocumentsGUI":

        self.state: TopicModelContainer = state
        self.n_topics = state.num_topics
        self.topic_id.value = 0
        self.topic_id.max = self.n_topics - 1

        self.prev_topic_id.on_click(self.goto_previous)
        self.next_topic_id.on_click(self.goto_next)

        self.topic_id.observe(self.update_handler, names='value')
        self.threshold.observe(self.update_handler, names='value')
        self.n_top.observe(self.update_handler, names='value')

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

    def update_handler(self, *_):

        topic_token_weights = self.state.inferred_topics.topic_token_weights
        document_topic_weights = self.state.inferred_topics.document_topic_weights
        document_index = self.state.inferred_topics.document_index

        self.output.clear_output()

        # if self.n_topics != state.num_topics:
        #    self.setup(n_topics)

        with self.output:

            self.text.value = tm.get_topic_title2(topic_token_weights, self.topic_id.value)

            topic_documents = tm.get_relevant_topic_documents(
                document_topic_weights=document_topic_weights,
                document_index=document_index,
                threshold=self.threshold.value,
                n_top=self.n_top.value,
                topic_id=self.topic_id.value,
            )

            if topic_documents is not None:
                display(topic_documents)


def display_gui(state: TopicModelContainer):
    gui: TopicDocumentsGUI = TopicDocumentsGUI().setup(state=state)
    display(gui.layout())
    gui.update_handler()
