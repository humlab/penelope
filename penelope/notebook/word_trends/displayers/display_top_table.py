import contextlib
from typing import Any

import IPython.display
import pandas as pd
from ipywidgets import HTML, Button, Dropdown, GridBox, HBox, Layout, Output, VBox
from penelope.corpus import VectorizedCorpus
from penelope.notebook.utility import create_js_download
from perspective import PerspectiveWidget

from .interface import ITrendDisplayer


# FIXME #72 Word trends: No data in top tokens displayer
class TopTokensDisplayer(ITrendDisplayer):
    def __init__(self, corpus: VectorizedCorpus = None, name: str = "TopTokens"):
        super().__init__(name=name)
        self.corpus: VectorizedCorpus = corpus

        self._top_count: Dropdown = Dropdown(
            options=[10 ** i for i in range(0, 7)],
            value=100,
            placeholder='Record count limit',
            layout=Layout(width='auto'),
        )
        self._save = Button(description='Save data', layout=Layout(width='auto'))
        self._download = Button(description='Download data', layout=Layout(width='auto'))
        self._download_output: Output = Output()
        self._table: PerspectiveWidget = None
        self._kind: Dropdown = Dropdown(
            options=['token', 'token/count', 'token+count'],
            value='token+count',
            description='',
            disabled=False,
            layout=Layout(width='100px'),
        )
        self._time_period: Dropdown = Dropdown(
            options=['year', 'lustrum', 'decade'],
            value='decade',
            description='',
            disabled=False,
            layout=Layout(width='100px'),
        )
        self.category_name = "time_period"

    def setup(self, *_, **__) -> "TopTokensDisplayer":
        self._table = PerspectiveWidget(self.data)
        self._download.on_click(self.download)
        self._top_count.observe(self.update, 'value')
        self._time_period.observe(self.update, 'value')
        self._kind.observe(self.update, 'value')
        return self

    def compile(self, corpus: VectorizedCorpus, **_) -> Any:  # pylint: disable=arguments-differ
        self.corpus = corpus
        # FIXME: #102 TopTokensDisplayer - Always group data from now on?
        if self.time_period != 'year':
            corpus = corpus.group_by_time_period(
                time_period_specifier=self.time_period, target_column_name=self.category_name
            )

        top_terms: pd.DataFrame = corpus.get_top_terms(
            category_column=self.category_name, n_count=self.top_count, kind=self.kind
        )
        return top_terms

    def plot(self, **_) -> "TopTokensDisplayer":  # pylint: disable=arguments-differ

        self.update()
        return self

    def update(self, *_):
        self._table.load(self.data)

    def download(self, *_):
        with contextlib.suppress(Exception):
            with self._download_output:
                js_download = create_js_download(self.data, index=True)
                if js_download is not None:
                    IPython.display.display(js_download)

    def layout(self) -> GridBox:
        layout: GridBox = GridBox(
            [
                HBox(
                    [
                        VBox([HTML("<b>Top count</b>"), self._top_count]),
                        VBox([HTML("<b>Grouping</b>"), self._time_period]),
                        VBox([HTML("<b>Kind</b>"), self._kind]),
                        VBox(
                            [
                                HTML('😢'),
                                self._download,
                            ]
                        ),
                        self._download_output,
                    ]
                ),
                self._table,
            ],
            layout=Layout(width='auto'),
        )
        return layout

    @property
    def data(self) -> pd.DataFrame:
        return self.compile(corpus=self.corpus)

    @property
    def top_count(self) -> int:
        return self._top_count.value

    @property
    def time_period(self) -> str:
        return self._time_period.value

    @property
    def kind(self) -> str:
        return self._kind.value
