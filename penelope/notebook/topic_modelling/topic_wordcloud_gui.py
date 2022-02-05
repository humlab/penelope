import ipywidgets as w
import pandas as pd
from IPython.display import display

import penelope.plot as plot_utility
import penelope.utility as utility
from penelope import topic_modelling as tm

from .. import grid_utility as gu
from . import mixins as mx
from . import model_container as mc

PLOT_OPTS = {'max_font_size': 100, 'background_color': 'white', 'width': 1200, 'height': 600}
OUTPUT_OPTIONS = ['Wordcloud', 'Table', 'CSV', 'XLSX', 'Clipboard']
CLEAR_OUTPUT: bool = True
view = w.Output()


@view.capture(clear_output=CLEAR_OUTPUT)
def display_wordcloud(
    inferred_topics: tm.InferredTopicsData,
    topic_id: int = 0,
    n_words: int = 100,
    output_format: str = 'Wordcloud',
):

    tokens = inferred_topics.get_topic_title(topic_id, n_tokens=n_words)

    if len(tokens) == 0:
        raise ValueError("No data! Please change selection.")

    if output_format == 'Wordcloud':
        plot_utility.plot_wordcloud(
            inferred_topics.get_topic_tokens(topic_id), token='token', weight='weight', max_words=n_words, **PLOT_OPTS
        )
    else:
        top_tokens: pd.DataFrame = inferred_topics.get_topic_top_tokens(topic_id=topic_id, n_tokens=n_words)
        g = gu.table_widget(top_tokens)
        display(g)
        if output_format.lower() in ('xlsx', 'csv', 'clipboard'):
            utility.ts_store(data=top_tokens, extension=output_format.lower(), basename='topic_top_tokens')


class WordcloudGUI(mx.NextPrevTopicMixIn, mx.AlertMixIn, mx.TopicsStateGui):
    def __init__(self, state: mc.TopicModelContainer):

        super().__init__(state=state)

        self.n_topics: int = self.inferred_n_topics

        self._text: w.HTML = w.HTML()
        self._word_count: w.IntSlider = w.IntSlider(
            description='#Words', min=5, max=250, step=1, value=75, continuous_update=False
        )
        self._output_format: w.Dropdown = w.Dropdown(
            description='Format', options=OUTPUT_OPTIONS, value=OUTPUT_OPTIONS[0], layout=dict(width="200px")
        )

    def setup(self, **kwargs) -> "WordcloudGUI":
        super().setup(**kwargs)
        self._topic_id.observe(self.update_handler, 'value')
        self._word_count.observe(self.update_handler, 'value')
        self._output_format.observe(self.update_handler, 'value')
        return self

    def update_handler(self, *_):

        self._topic_id.unobserve(self.update_handler, 'value')

        if self.n_topics != self.inferred_n_topics:
            self.n_topics = self.inferred_n_topics
            self.topic_id = (0, self.inferred_n_topics - 1, self.inferred_topics.topic_labels)

        try:
            self.alert("⌛Computing...")
            display_wordcloud(
                inferred_topics=self.inferred_topics,
                topic_id=self.topic_id,
                n_words=self._word_count.value,
                output_format=self._output_format.value,
            )
            self.alert("✅ Done!")
        except Exception as ex:
            self.warn(f"😡 {ex}")
        finally:
            self._topic_id.observe(self.update_handler, 'value')
        return self

    def layout(self) -> w.VBox:
        return w.VBox(
            [
                self._text,
                w.HBox([self._next_prev_layout, self._word_count, self._output_format, self._alert]),
                view,
            ]
        )


def display_gui(state: mc.TopicModelContainer):

    gui: WordcloudGUI = WordcloudGUI(state).setup()
    display(gui.layout())
    gui.update_handler()

    return gui
