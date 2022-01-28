from __future__ import annotations

import uuid

import ipywidgets as w
from IPython.display import display

from penelope import utility as pu

from . import mixins as mx
from .model_container import TopicModelContainer


class BrowseTopicDocumentsGUI(mx.NextPrevTopicMixIn, mx.AlertMixIn, mx.TopicsStateGui):
    def __init__(self, state: TopicModelContainer | dict):
        super().__init__()

        self.state: TopicModelContainer | dict = state

        timespan: tuple[int, int] = self.inferred_topics.year_period
        self.n_topics: int = self.inferred_n_topics

        self.text_id: str = str(uuid.uuid4())[:6]

        self._year_range: w.IntRangeSlider = w.IntRangeSlider(min=timespan[0], max=timespan[1], value=timespan)
        self._threshold: w.FloatSlider = w.FloatSlider(min=0.01, max=1.0, value=0.05, step=0.01)
        self._text: w.HTML = w.HTML(value=f"<span class='{self.text_id}'></span>")
        self._max_n_top: w.IntSlider = w.IntSlider(min=1, max=50000, value=500, disabled=False)
        self._output: w.Output = w.Output()

    def setup(self, **kwargs) -> "BrowseTopicDocumentsGUI":  # pylint: disable=arguments-differ
        super().setup(**kwargs)

        self._topic_id.value = 0
        self._topic_id.max = self.n_topics - 1
        self._topic_id.observe(self.update_handler, names='value')

        self._threshold.observe(self.update_handler, names='value')
        self._max_n_top.observe(self.update_handler, names='value')

        return self

    def layout(self):
        return w.VBox(
            [
                w.HBox(
                    [
                        w.VBox(
                            [
                                w.HBox([self._prev_topic_id, self._next_topic_id]),
                                w.HTML("<b>Threshold</b> (topic's weight in doc)"),
                                self._threshold,
                                w.HTML("<b>Max result count</b>"),
                                self._max_n_top,
                            ]
                        ),
                        w.VBox([self._topic_id]),
                    ]
                ),
                self._text,
                self._output,
            ]
        )

    @property
    def filter_opts(self) -> pu.PropertyValueMaskingOpts:
        return pu.PropertyValueMaskingOpts()

    def update_handler(self, *_):

        self._output.clear_output()

        with self._output:

            self._text.value = self.inferred_topics.get_topic_title2(self.topic_id)

            topic_documents = (
                self.inferred_topics.calculator.reset()
                .filter_by_data_keys(topic_id=self.topic_id)
                .threshold(threshold=self.threshold)
                .filter_by_document_keys(**(self.filter_opts.data))
                .filter_by_n_top(n_top=self.n_top)
                .value
            )

            if topic_documents is not None:
                display(topic_documents)

    @property
    def threshold(self) -> float:
        return self._threshold.value

    @property
    def n_top(self) -> int:
        return self._max_n_top.value


def display_gui(state: TopicModelContainer):
    gui: BrowseTopicDocumentsGUI = BrowseTopicDocumentsGUI(state=state).setup()
    display(gui.layout())
    gui.update_handler()
