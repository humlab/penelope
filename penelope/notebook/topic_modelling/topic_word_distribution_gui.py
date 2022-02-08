import warnings

import bokeh
import bokeh.plotting
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import Dropdown, HBox, IntSlider, Output, VBox  # type: ignore

from penelope import utility

from . import mixins as mx
from . import model_container as mc

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


def display_topic_tokens(top_tokens: pd.DataFrame, n_words: int = 100, output_format: str = 'Chart'):

    if len(top_tokens) == 0:
        print("No data! Please change selection.")
        return

    top_tokens: pd.DataFrame = (
        top_tokens.copy()
        .drop('topic_id', axis=1)
        .assign(weight=lambda x: 100.0 * x.weight)
        .sort_values('weight', axis=0, ascending=False)
        .reset_index()
        .head(n_words)
    )
    if output_format.lower() == 'chart':
        top_tokens = top_tokens.assign(xs=top_tokens.index, ys=top_tokens.weight)
        p = plot_topic_word_distribution(
            top_tokens, plot_width=1200, plot_height=500, title='', tools='box_zoom,wheel_zoom,pan,reset'
        )
        bokeh.plotting.show(p)
    elif output_format.lower() in ('xlsx', 'csv', 'clipboard'):
        utility.ts_store(data=top_tokens, extension=output_format.lower(), basename='topic_word_distribution')
    else:
        display(top_tokens)


OUTPUT_OPTIONS = ['Chart', 'XLSX', 'CSV', 'Clipboard', 'Table']


class TopicWordDistributionGUI(mx.NextPrevTopicMixIn, mx.TopicsStateGui):
    def __init__(self, state: mc.TopicModelContainer):

        super().__init__(state=state)

        self.n_topics: int = self.inferred_n_topics
        self._n_words: IntSlider = IntSlider(description='#Words', min=5, max=500, step=1, value=75)
        self._output_format: Dropdown = Dropdown(
            description='Format', options=OUTPUT_OPTIONS, value=OUTPUT_OPTIONS[0], layout=dict(width="200px")
        )
        self._output: Output = Output()

    def setup(self, **kwargs) -> "TopicWordDistributionGUI":
        super().setup(**kwargs)
        self._n_words.observe(self.update_handler, 'value')
        self._output_format.observe(self.update_handler, 'value')
        self.topic_id = (0, self.inferred_n_topics - 1, self.inferred_topics.topic_labels)
        return self

    def update_handler(self, *_):

        if self.n_topics != self.inferred_n_topics:
            self.n_topics = self.inferred_n_topics
            self.topic_id = (0, self.inferred_n_topics - 1, self.inferred_topics.topic_labels)

        self.buzy(True)
        top_tokens: pd.DataFrame = self.inferred_topics.get_topic_top_tokens(
            topic_id=self.topic_id, n_tokens=self.n_words
        )
        display_topic_tokens(
            top_tokens=top_tokens,
            n_words=self.n_words,
            output_format=self.output_format,
        )
        self.buzy(False)

    def buzy(self, value: bool = False) -> None:
        self._topic_id.disabled = value
        self._n_words.disabled = value
        self._output_format.disabled = value

    def layout(self) -> VBox:
        return VBox(
            [
                HBox([self._next_prev_layout, self._n_words, self._output_format]),
                self._output,
            ]
        )

    @property
    def n_words(self) -> int:
        return self._n_words.value

    @property
    def output_format(self) -> int:
        return self._output_format.value


def display_gui(state: mc.TopicModelContainer) -> None:
    gui = TopicWordDistributionGUI(state).setup()
    display(gui.layout())
    gui.update_handler()
