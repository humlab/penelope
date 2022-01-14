import pandas as pd
import penelope.plot as plot_utility
import penelope.utility as utility
from IPython.display import display
from ipywidgets import HTML, Button, Dropdown, HBox, IntProgress, IntSlider, Output, VBox  # type: ignore
from penelope import topic_modelling as tm

from .. import widgets_utils
from .model_container import TopicModelContainer

pd.set_option("display.max_rows", None)
pd.set_option('max_colwidth', 200)

PLOT_OPTS = {'max_font_size': 100, 'background_color': 'white', 'width': 1200, 'height': 600}

TEXT_ID: str = 'tx02'
OUTPUT_OPTIONS = ['Wordcloud', 'Table', 'CSV', 'XLSX', 'Clipboard']
CLEAR_OUTPUT: bool = True
view = Output()


@view.capture(clear_output=CLEAR_OUTPUT)
def display_wordcloud(
    inferred_topics: tm.InferredTopicsData,
    topic_id: int = 0,
    n_words: int = 100,
    output_format: str = 'Wordcloud',
    gui: "WordcloudGUI" = None,
):

    gui.tick(1)

    try:
        tokens = tm.get_topic_title(inferred_topics.topic_token_weights, topic_id, n_tokens=n_words)

        if len(tokens) == 0:
            print("No data! Please change selection.")
            return

        gui.text.value = 'ID {}: {}'.format(topic_id, tokens)

        gui.tick()

        if output_format == 'Wordcloud':
            plot_utility.plot_wordcloud(
                inferred_topics.topic_token_weights.loc[inferred_topics.topic_token_weights.topic_id == topic_id],
                token='token',
                weight='weight',
                max_words=n_words,
                **PLOT_OPTS,
            )
        else:
            topic_top_tokens: pd.DataFrame = tm.get_topic_top_tokens(
                inferred_topics.topic_token_weights, topic_id=topic_id, n_tokens=n_words
            )
            if output_format == 'Table':
                display(topic_top_tokens)
            elif output_format.lower() in ('xlsx', 'csv', 'clipboard'):
                utility.ts_store(data=topic_top_tokens, extension=output_format.lower(), basename='topic_top_tokens')

    except IndexError:
        print('No data for topic')
    gui.tick(0)


class WordcloudGUI:
    def __init__(self, state: TopicModelContainer):

        self.state: TopicModelContainer = state
        self.n_topics: int = state.num_topics
        self.text_id: str = TEXT_ID
        self.text: HTML = HTML(value=f"<span class='{TEXT_ID}'></span>", placeholder='', description='')
        self.topic_id: IntSlider = IntSlider(
            description='Topic ID', min=0, max=state.num_topics - 1, step=1, value=0, continuous_update=False
        )
        self.word_count: IntSlider = IntSlider(
            description='#Words', min=5, max=250, step=1, value=75, continuous_update=False
        )
        self.output_format: Dropdown = Dropdown(
            description='Format', options=OUTPUT_OPTIONS, value=OUTPUT_OPTIONS[0], layout=dict(width="200px")
        )
        self.progress: IntProgress = IntProgress(min=0, max=4, step=1, value=0, layout=dict(width="95%"))
        # self.output: Output = Output()
        self.prev_topic_id: Button = None
        self.next_topic_id: Button = None

    def setup(self) -> "WordcloudGUI":

        self.prev_topic_id = widgets_utils.button_with_previous_callback(self, 'topic_id', self.state.num_topics)
        self.next_topic_id = widgets_utils.button_with_next_callback(self, 'topic_id', self.state.num_topics)

        self.topic_id.observe(self.update_handler, 'value')
        self.word_count.observe(self.update_handler, 'value')
        self.output_format.observe(self.update_handler, 'value')

        return self

    def tick(self, n=None):
        self.progress.value = (self.progress.value + 1) if n is None else n

    def update_handler(self, *_):

        if self.n_topics != self.state.num_topics:
            self.n_topics = self.state.num_topics
            self.topic_id.value = 0
            self.topic_id.max = self.state.num_topics - 1

        display_wordcloud(
            inferred_topics=self.state.inferred_topics,
            topic_id=self.topic_id.value,
            n_words=self.word_count.value,
            output_format=self.output_format.value,
            gui=self,
        )

        return self

    def layout(self) -> VBox:
        return VBox(
            [
                self.text,
                HBox([self.prev_topic_id, self.next_topic_id, self.topic_id, self.word_count, self.output_format]),
                self.progress,
                view,
            ]
        )


def display_gui(state: TopicModelContainer):

    gui: WordcloudGUI = WordcloudGUI(state).setup()
    display(gui.layout())
    gui.update_handler()

    return gui
