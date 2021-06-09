import contextlib
from typing import Any

import IPython.display
import pandas as pd
from ipywidgets import HTML, Button, Dropdown, GridBox, HBox, Layout, Output, VBox
from penelope.co_occurrence.bundle import Bundle
from penelope.common.keyness import KeynessMetric
from penelope.corpus import VectorizedCorpus
from penelope.notebook.utility import create_js_download
from perspective import PerspectiveWidget

from .interface import ITrendDisplayer


# FIXME #72 Word trends: No data in top tokens displayer
class TopTokensDisplayer(ITrendDisplayer):
    def __init__(self, corpus: VectorizedCorpus = None, name: str = "TopTokens"):
        super().__init__(name=name)

        self.corpus: VectorizedCorpus = corpus

        self.keyness_options = {
            "TF": KeynessMetric.TF,
            "TF (norm)": KeynessMetric.TF_normalized,
            "TF-IDF": KeynessMetric.TF_IDF,
        }

        self._keyness: Dropdown = None
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
        self._keyness: Dropdown = Dropdown(
            options=self.keyness_options,
            value=KeynessMetric.TF,
            layout=Layout(width='auto'),
        )

        self._table = PerspectiveWidget(self.data)
        self._download.on_click(self.download)
        self._top_count.observe(self.load, 'value')
        self._time_period.observe(self.load, 'value')
        self._keyness.observe(self.load, 'value')
        self._kind.observe(self.load, 'value')
        return self

    def transform(self) -> VectorizedCorpus:
        corpus = self.corpus.group_by_time_period(
            time_period_specifier=self.time_period, target_column_name=self.category_name
        )
        return corpus

    def compile(self, **_) -> Any:  # pylint: disable=arguments-differ
        top_terms: pd.DataFrame = self.transform().get_top_terms(
            category_column=self.category_name, n_count=self.top_count, kind=self.kind
        )
        return top_terms

    def plot(self, **_) -> "TopTokensDisplayer":  # pylint: disable=arguments-differ

        self.load()
        return self

    def load(self, *_):
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
                        VBox([HTML("<b>Keyness</b>"), self._keyness]),
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
    def keyness(self) -> KeynessMetric:
        return self._keyness.value

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


class CoOccurrenceTopTokensDisplayer(TopTokensDisplayer):
    def __init__(self, bundle: Bundle, name: str = "TopTokens"):
        super().__init__(corpus=bundle.corpus, name=name)

        self.bundle = bundle
        self.keyness_options.update(
            {
                "HAL CWR": KeynessMetric.HAL_cwr,
                "PPMI": KeynessMetric.PPMI,
                "LLR": KeynessMetric.LLR,
                "LLR(D)": KeynessMetric.LLR_Dunning,
                "DICE": KeynessMetric.DICE,
            }
        )

    def transform(self) -> VectorizedCorpus:
        corpus: VectorizedCorpus = self.bundle.to_keyness_corpus(
            period_pivot=self.time_period,
            keyness=self.keyness,
            fill_gaps=False,
            normalize=False,
            global_threshold=1,  # FIXME Add
            pivot_column_name=self.category_name,
        )
        return corpus

    def plot(self, **_):
        self.data = self.compile()
        self.load()
