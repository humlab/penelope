import contextlib
from dataclasses import dataclass, field

import IPython.display
import pandas as pd
from ipywidgets import HTML, Button, Dropdown, GridBox, HBox, Layout, Output, VBox
from penelope.corpus.dtm import VectorizedCorpus
from penelope.notebook.utility import create_js_download
from perspective import PerspectiveWidget

from ._displayer import ITrendDisplayer


# FIXME #72 Word trends: No data in top tokens displayer
@dataclass
class TopTokensDisplayer(ITrendDisplayer):

    name: str = field(default="TopTokens")
    corpus: VectorizedCorpus = None

    _download_output: Output = Output()
    _top_count: Dropdown = Dropdown(
        options=[10 ** i for i in range(0, 7)],
        value=100,
        placeholder='Record count limit',
        layout=Layout(width='auto'),
    )
    _save = Button(description='Save data', layout=Layout(width='auto'))
    _download = Button(description='Download data', layout=Layout(width='auto'))
    _table: PerspectiveWidget = None
    _kind: Dropdown = Dropdown(
        options=['token', 'token/count', 'token+count'],
        value='token+count',
        description='',
        disabled=False,
        layout=Layout(width='100px'),
    )
    _category: Dropdown = Dropdown(
        options=['year', 'lustrum', 'decade'],
        value='decade',
        description='',
        disabled=False,
        layout=Layout(width='100px'),
    )

    def setup(self, *_, **__) -> "TopTokensDisplayer":
        self._table = PerspectiveWidget(self.data)
        self._download.on_click(self.download)
        self._top_count.observe(self.update, 'value')
        self._category.observe(self.update, 'value')
        self._kind.observe(self.update, 'value')
        return self

    def compile(self, *, corpus: VectorizedCorpus, **__) -> pd.DataFrame:
        self.corpus = corpus

        if self.category != 'year':
            corpus = corpus.group_by_period(period=self.category)

        top_terms: pd.DataFrame = corpus.get_top_terms(
            category_column='category', n_count=self.top_count, kind=self.kind
        )
        return top_terms

    def plot(self, plot_data: dict, **_) -> "TopTokensDisplayer":  # pylint: disable=unused-argument

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
                        VBox([HTML("<b>Grouping</b>"), self._category]),
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
    def category(self) -> str:
        return self._category.value

    @property
    def kind(self) -> str:
        return self._kind.value
