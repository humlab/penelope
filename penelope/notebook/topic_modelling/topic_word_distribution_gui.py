import contextlib
import warnings

import bokeh
import bokeh.plotting
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import HTML, Button, Dropdown, HBox, IntSlider, Output, VBox  # type: ignore
from penelope import topic_modelling, utility

from .. import widgets_utils
from .model_container import TopicModelContainer

warnings.filterwarnings("ignore", category=DeprecationWarning)


def plot_topic_word_distribution(tokens: pd.DataFrame, **args):

    source = bokeh.models.ColumnDataSource(tokens)

    p = bokeh.plotting.figure(toolbar_location="right", **args)

    _ = p.circle(x='xs', y='ys', source=source)

    label_style = dict(level='overlay', text_font_size='8pt', angle=np.pi / 6.0)

    text_aligns = ['left', 'right']
    for i in [0, 1]:
        label_source = bokeh.models.ColumnDataSource(tokens.iloc[i::2])
        labels = bokeh.models.LabelSet(
            x='xs',
            y='ys',
            text_align=text_aligns[i],
            text='token',
            text_baseline='middle',
            y_offset=5 * (1 if i == 0 else -1),
            x_offset=5 * (1 if i == 0 else -1),
            source=label_source,
            **label_style,
        )
        p.add_layout(labels)

    p.xaxis[0].axis_label = 'Token #'
    p.yaxis[0].axis_label = 'Probability%'
    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "6pt"
    p.axis.major_label_standoff = 0
    return p


def display_topic_tokens(
    topic_token_weights: pd.DataFrame, topic_id: int = 0, n_words: int = 100, output_format: str = 'Chart'
):

    tokens: pd.DataFrame = (
        topic_modelling.get_topic_top_tokens(topic_token_weights, topic_id=topic_id, n_tokens=n_words)
        .copy()
        .drop('topic_id', axis=1)
        .assign(weight=lambda x: 100.0 * x.weight)
        .sort_values('weight', axis=0, ascending=False)
        .reset_index()
        .head(n_words)
    )

    if len(tokens) == 0:
        print("No data! Please change selection.")
        return

    if output_format.lower() == 'chart':
        tokens = tokens.assign(xs=tokens.index, ys=tokens.weight)
        p = plot_topic_word_distribution(
            tokens, plot_width=1200, plot_height=500, title='', tools='box_zoom,wheel_zoom,pan,reset'
        )
        bokeh.plotting.show(p)
    elif output_format.lower() in ('xlsx', 'csv', 'clipboard'):
        utility.ts_store(data=tokens, extension=output_format.lower(), basename='topic_word_distribution')
    else:
        display(tokens)


TEXT_ID: str = 'wc01'
OUTPUT_OPTIONS = ['Chart', 'XLSX', 'CSV', 'Clipboard', 'Table']


class TopicWordDistributionGUI:
    def __init__(self, state: TopicModelContainer):

        self.state: TopicModelContainer = state
        self.n_topics: int = state.num_topics
        self.text_id: str = TEXT_ID
        self.text: HTML = widgets_utils.text_widget(TEXT_ID)
        self.topic_id: IntSlider = IntSlider(description='Topic ID', min=0, max=state.num_topics - 1, step=1, value=0)
        self.n_words: IntSlider = IntSlider(description='#Words', min=5, max=500, step=1, value=75)
        self.output_format: Dropdown = Dropdown(
            description='Format', options=OUTPUT_OPTIONS, value=OUTPUT_OPTIONS[0], layout=dict(width="200px")
        )
        self.prev_topic_id: Button = None
        self.next_topic_id: Button = None
        self.output: Output = Output()

    def setup(self) -> "TopicWordDistributionGUI":

        self.prev_topic_id = widgets_utils.button_with_previous_callback(self, 'topic_id', self.state.num_topics)
        self.next_topic_id = widgets_utils.button_with_next_callback(self, 'topic_id', self.state.num_topics)

        self.topic_id.observe(self.update_handler, 'value')
        self.n_words.observe(self.update_handler, 'value')
        self.output_format.observe(self.update_handler, 'value')

        return self

    def update_handler(self, *_):

        if self.n_topics != self.state.num_topics:
            self.n_topics = self.state.num_topics
            self.topic_id.value = 0
            self.topic_id.max = self.state.num_topics - 1

        self.buzy(True)
        with contextlib.suppress(Exception):
            display_topic_tokens(
                topic_token_weights=self.state.inferred_topics.topic_token_weights,
                topic_id=self.topic_id.value,
                n_words=self.n_words.value,
                output_format=self.output_format.value,
            )
        self.buzy(False)

    def buzy(self, value: bool = False) -> None:
        self.topic_id.disabled = value
        self.n_words.disabled = value
        self.output_format.disabled = value

    def layout(self) -> VBox:
        return VBox(
            [
                self.text,
                HBox([self.prev_topic_id, self.next_topic_id, self.topic_id, self.n_words, self.output_format]),
                self.output,
            ]
        )


def display_gui(state: TopicModelContainer) -> None:
    gui = TopicWordDistributionGUI(state).setup()
    display(gui.layout())
    gui.update_handler()
