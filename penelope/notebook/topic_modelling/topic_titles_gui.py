import abc

import ipywidgets as w
import pandas as pd
from IPython.display import Javascript
from IPython.display import display as IPython_display

from penelope.topic_modelling import filter_topic_tokens_overview

from ..utility import create_js_download

pd.options.mode.chained_assignment = None


class TopicTitlesGUI(abc.ABC):
    def __init__(self, topics: pd.DataFrame, n_tokens: int = 500):

        self.n_tokens: int = n_tokens
        self.topics: pd.DataFrame = topics
        self.reduced_topics: pd.DataFrame = None

        self._count_slider: w.IntSlider = w.IntSlider(
            description="Tokens", min=1, max=n_tokens, value=50, continuous_update=False, layout=dict(width="40%")
        )
        self._search_text: w.Text = w.Text(
            description="Find", placeholder="(enter at least three characters", layout=dict(width="40%")
        )
        self._download_button: w.Button = w.Button(description="Download", layout=dict(width='100px'))
        self._prune_tokens: w.ToggleButton = w.ToggleButton(
            description="Prune", icon="check", value=True, layout=dict(width='100px')
        )
        self._output: w.Output = w.Output()
        self.js_download: Javascript = None

    def setup(self) -> "TopicTitlesGUI":
        self._count_slider.observe(self._update, "value")
        self._search_text.observe(self._update, "value")
        self._prune_tokens.observe(self._update, "value")
        self._download_button.on_click(self.download)
        self._prune_tokens.observe(self._toggle_state_changed, 'value')
        self._toggle_state_changed(dict(owner=self._prune_tokens))
        return self

    def _toggle_state_changed(self, event):
        event['owner'].icon = 'check' if event['owner'].value else ''

    def layout(self):
        return w.VBox(
            (
                w.HBox(
                    (
                        self._count_slider,
                        self._download_button,
                    )
                ),
                w.HBox(
                    (
                        self._search_text,
                        self._prune_tokens,
                    )
                ),
                self._output,
            )
        )

    def download(self, *_):
        with self._output:
            js_download = create_js_download(self.reduced_topics, index=True)
            if js_download is not None:
                IPython_display(js_download)

    def _update(self, *_):
        self._output.clear_output()
        with self._output:
            if len(self._search_text.value) > 2:
                self.reduce_topics()
                self.update()

    def reduce_topics(self):
        self.reduced_topics = filter_topic_tokens_overview(
            self.topics,
            search_text=self._search_text.value,
            n_top=self._count_slider.value,
            truncate_tokens=self._prune_tokens.value,
        )

    def update(self) -> None:
        IPython_display(self.reduced_topics)


class PandasTopicTitlesGUI(TopicTitlesGUI):

    PANDAS_TABLE_STYLE = [
        dict(
            selector="th",
            props=[
                ('font-size', '11px'),
                ('text-align', 'left'),
                ('font-weight', 'bold'),
                ('color', '#6d6d6d'),
                ('background-color', '#f7f7f9'),
            ],
        ),
        dict(
            selector="td",
            props=[
                ('font-size', '11px'),
                ('text-align', 'left'),
            ],
        ),
    ]

    def update(self, *_):
        styled_reduced_topics = self.reduced_topics.style.set_table_styles(self.PANDAS_TABLE_STYLE)
        IPython_display(styled_reduced_topics)

    def setup(self) -> "TopicTitlesGUI":
        super().setup()
        pd.set_option('colheader_justify', 'left')
        return self
