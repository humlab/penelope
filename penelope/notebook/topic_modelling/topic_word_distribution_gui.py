import warnings

import bokeh
import bokeh.plotting
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import Dropdown, HBox, IntSlider, Output, VBox  # type: ignore

from penelope import utility

from .. import grid_utility as gu
from . import mixins as mx
from . import model_container as mc

warnings.filterwarnings("ignore", category=DeprecationWarning)


def plot_topic_word_distribution(tokens: pd.DataFrame, **args):

    source = bokeh.models.ColumnDataSource(tokens)

    p = bokeh.plotting.figure(toolbar_location="right", sizing_mode='scale_width', **args)
    p.left[0].formatter.use_scientific = False  # pylint: disable=unsubscriptable-object

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


OUTPUT_OPTIONS = ['Chart', 'XLSX', 'CSV', 'Clipboard', 'Table']


class TopicWordDistributionGUI(mx.AlertMixIn, mx.NextPrevTopicMixIn, mx.TopicsStateGui):
    def __init__(self, state: mc.TopicModelContainer):

        super().__init__(state=state)
        self.data: pd.DataFrame = None
        self.n_topics: int = self.inferred_n_topics
        self._n_words: IntSlider = IntSlider(description='', min=5, max=500, step=1, value=75)
        self._output_format: Dropdown = Dropdown(
            options=OUTPUT_OPTIONS, value=OUTPUT_OPTIONS[0], layout=dict(width="200px")
        )
        self._output: Output = Output()

    def setup(self, **kwargs) -> "TopicWordDistributionGUI":
        super().setup(**kwargs)
        self._n_words.observe(self.update_handler, 'value')
        self._output_format.observe(self.update_handler, 'value')
        self._topic_id.observe(self.update_handler, 'value')
        self.topic_id = (0, self.inferred_n_topics - 1, self.inferred_topics.topic_labels)
        return self

    def compute(self) -> pd.DataFrame:
        data: pd.DataFrame = (
            self.inferred_topics.get_topic_top_tokens(topic_id=self.topic_id, n_tokens=self.n_words)
            .copy()
            .drop('topic_id', axis=1)
            .assign(weight=lambda x: 100.0 * x.weight)
            .sort_values('weight', axis=0, ascending=False)
            .reset_index()
            .head(self.n_words)
        )
        return data

    def update_handler(self, *_):

        if self.n_topics != self.inferred_n_topics:
            self.n_topics = self.inferred_n_topics
            self.topic_id = (0, self.inferred_n_topics - 1, self.inferred_topics.topic_labels)

        self.buzy(True)
        self.data: pd.DataFrame = self.compute()
        self.display_handler()
        self.buzy(False)

    def display_handler(self, *_):

        if len(self.data) == 0:
            self.alert("No data!")
            return

        self._output.clear_output()
        with self._output:
            if self.output_format.lower() == 'chart':
                top_tokens: pd.DataFrame = self.data.assign(xs=self.data.index, ys=self.data.weight)
                p = plot_topic_word_distribution(
                    top_tokens, width=1200, height=500, title='', tools='box_zoom,wheel_zoom,pan,reset'
                )
                bokeh.plotting.show(p)
            else:
                if self.output_format.lower() in ('xlsx', 'csv', 'clipboard'):
                    utility.ts_store(
                        data=self.data, extension=self.output_format.lower(), basename='topic_word_distribution'
                    )
                g: gu.TableWidget = gu.table_widget(self.data)
                display(g)
        self.alert("✅")

    def buzy(self, value: bool = False) -> None:
        self._topic_id.disabled = value
        self._n_words.disabled = value
        self._output_format.disabled = value

    def layout(self) -> VBox:
        return VBox(
            [
                HBox([self._next_prev_layout, self._n_words, self._output_format, self._alert]),
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
