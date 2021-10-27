import warnings

import bokeh
import bokeh.plotting
import numpy as np
import pandas as pd
import penelope.topic_modelling as topic_modelling
from IPython.display import display
from ipywidgets import HTML, Dropdown, HBox, IntProgress, IntSlider, VBox, fixed, interactive  # type: ignore

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
    state: TopicModelContainer, topic_id: int = 0, n_words: int = 100, output_format: str = 'Chart', gui=None
):
    def tick(n=None):
        if gui is not None:
            gui.progress.value = (gui.progress.value + 1) if n is None else n

    if gui is not None and gui.n_topics != state.num_topics:
        gui.n_topics = state.num_topics
        gui.topic_id.value = 0
        gui.topic_id.max = state.num_topics - 1

    tick(1)

    tokens = (
        topic_modelling.get_topic_top_tokens(
            state.inferred_topics.topic_token_weights, topic_id=topic_id, n_tokens=n_words
        )
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

    if output_format == 'Chart':
        tick()
        tokens = tokens.assign(xs=tokens.index, ys=tokens.weight)
        p = plot_topic_word_distribution(
            tokens, plot_width=1200, plot_height=500, title='', tools='box_zoom,wheel_zoom,pan,reset'
        )
        bokeh.plotting.show(p)
        tick()
    else:
        display(tokens)

    tick(0)


TEXT_ID: str = 'wc01'
OUTPUT_OPTIONS = ['Chart', 'Table']


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
        self.progress: IntProgress = IntProgress(min=0, max=4, step=1, value=0, layout=dict(width="95%"))
        self.prev_topic_id = None
        self.next_topic_id = None

        self.iw = None

    def setup(self) -> "TopicWordDistributionGUI":

        self.prev_topic_id = widgets_utils.button_with_previous_callback(self, 'topic_id', self.state.num_topics)
        self.next_topic_id = widgets_utils.button_with_next_callback(self, 'topic_id', self.state.num_topics)

        self.iw = interactive(
            display_topic_tokens,
            state=fixed(self.state),
            topic_id=self.topic_id,
            n_words=self.n_words,
            output_format=self.output_format,
            self=fixed(self),
        )

        return self

    def layout(self) -> VBox:
        return VBox(
            [
                self.text,
                HBox([self.prev_topic_id, self.next_topic_id, self.topic_id, self.n_words, self.output_format]),
                self.progress,
                self.iw.children[-1],
            ]
        )


def display_gui(state: TopicModelContainer):

    gui = TopicWordDistributionGUI(state).setup()

    display(gui.layout())

    gui.iw.update()
