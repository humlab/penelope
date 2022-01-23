import uuid
import warnings

from IPython.display import display
from ipywidgets import HTML, Button, FloatSlider, HBox, IntSlider, Label, Output, VBox  # type: ignore

from .model_container import TopicModelContainer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TopicDocumentsGUI:
    def __init__(self):

        self.n_topics: int = 0
        self.text_id: str = str(uuid.uuid4())[:6]
        self.state: TopicModelContainer = None

        self._text: HTML = HTML(value=f"<span class='{self.text_id}'></span>")
        self._topic_id: IntSlider = IntSlider(
            description='Topic ID', min=0, max=199, step=1, value=0, continuous_update=False
        )
        self._n_top: IntSlider = IntSlider(description='', min=5, max=500, step=1, value=75)
        self._threshold = FloatSlider(
            description='Threshold', min=0.0, max=1.0, step=0.01, value=0.20, continues_update=False
        )
        self._output: Output = Output()

        button_style: dict = dict(description_width='initial', button_color='lightgreen')

        self._prev_topic_id: Button = Button(description="<<", layout=button_style)
        self._next_topic_id: Button = Button(description=">>", layout=button_style)

    def setup(self, *, state: TopicModelContainer) -> "TopicDocumentsGUI":

        self.state: TopicModelContainer = state
        self.n_topics = state.num_topics
        self._topic_id.value = 0
        self._topic_id.max = self.n_topics - 1

        self._prev_topic_id.on_click(self.goto_previous)
        self._next_topic_id.on_click(self.goto_next)

        self._topic_id.observe(self.update_handler, names='value')
        self._threshold.observe(self.update_handler, names='value')
        self._n_top.observe(self.update_handler, names='value')

        return self

    def goto_previous(self, *_):
        self._topic_id.value = (self._topic_id.value - 1) % self._topic_id.max

    def goto_next(self, *_):
        self._topic_id.value = (self._topic_id.value + 1) % self._topic_id.max

    def layout(self):
        return VBox(
            [
                HBox(
                    [
                        VBox(
                            [
                                HBox([self._prev_topic_id, self._next_topic_id]),
                                Label("Max number of documents to show"),
                                self._n_top,
                            ]
                        ),
                        VBox([self._topic_id, self._threshold]),
                    ]
                ),
                self._text,
                self._output,
            ]
        )

    def update_handler(self, *_):

        self._output.clear_output()

        # if self.n_topics != state.num_topics:
        #    self.setup(n_topics)

        with self._output:

            self._text.value = self.state.inferred_topics.get_topic_title2(self.topic_id)

            topic_documents = (
                self.state.inferred_topics.calculator.reset()
                .filter_by_data_keys(topic_id=self.topic_id)
                .threshold(threshold=self.threshold)
                .filter_by_n_top(n_top=self.n_top)
            )

            if topic_documents is not None:
                display(topic_documents)

    @property
    def threshold(self) -> float:
        return self._threshold.value

    @property
    def n_top(self) -> int:
        return self._n_top.value

    @property
    def topic_id(self) -> int:
        return self._topic_id.value


def display_gui(state: TopicModelContainer):
    gui: TopicDocumentsGUI = TopicDocumentsGUI().setup(state=state)
    display(gui.layout())
    gui.update_handler()
