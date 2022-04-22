import contextlib
from typing import Any, Sequence

import IPython.display
import pandas as pd
from IPython.display import display
from ipywidgets import HTML, Button, Dropdown, GridBox, HBox, Layout, Output, VBox

from penelope.co_occurrence.bundle import Bundle
from penelope.co_occurrence.keyness import ComputeKeynessOpts
from penelope.common.keyness import KeynessMetric, KeynessMetricSource
from penelope.corpus import VectorizedCorpus

from ... import grid_utility as gu
from ... import utility as nu
from .interface import ITrendDisplayer

# pylint: disable=too-many-instance-attributes


class TopTokensDisplayer(ITrendDisplayer):
    def __init__(self, corpus: VectorizedCorpus = None, name: str = "TopTokens", **opts):
        super().__init__(name=name, **opts)

        self.simple_display: bool = False

        self.corpus: VectorizedCorpus = corpus
        self._keyness: Dropdown = self.keyness_widget()
        self._placeholder: VBox = VBox()
        self._top_count: Dropdown = Dropdown(
            options=[10**i for i in range(0, 7)],
            value=100,
            placeholder='Record count limit',
            layout=Layout(width='auto'),
        )
        self._compute: Button = Button(description="Compute", button_style='success', layout=Layout(width='auto'))
        self._save = Button(description='Save data', layout=Layout(width='auto'))
        self._download = Button(description='Download data', layout=Layout(width='auto'))
        self._download_output: Output = Output()
        self._alert: HTML = HTML('.')
        self._kind: Dropdown = Dropdown(
            options=['token', 'token/count', 'token+count'],
            value='token+count',
            description='',
            disabled=False,
            layout=Layout(width='100px'),
        )
        self._temporal_key: Dropdown = Dropdown(
            options=['year', 'lustrum', 'decade'],
            value='decade',
            description='',
            disabled=False,
            layout=Layout(width='100px'),
        )
        self.category_name = "time_period"
        self._table: gu.DataGridOutput = None
        self._output: Output = None

    def keyness_widget(self) -> Dropdown:
        return Dropdown(
            options={
                "TF": KeynessMetric.TF,
                "TF (norm)": KeynessMetric.TF_normalized,
                "TF-IDF": KeynessMetric.TF_IDF,
            },
            value=KeynessMetric.TF,
            layout=Layout(width='auto'),
        )

    def start_observe(self):

        with contextlib.suppress(Exception):
            self._download.on_click(self.download, remove=True)

        self._download.on_click(self.download)

        with contextlib.suppress(Exception):
            self._compute.on_click(self.load, remove=True)

        self._compute.on_click(self.load)

        return self

    def setup(self, *_, **__) -> "TopTokensDisplayer":

        self._table = gu.DataGridOutput() if not self.simple_display else None
        self._output = Output() if self.simple_display else None

        self.start_observe()

        return self

    def transform(self) -> VectorizedCorpus:
        self.set_buzy(True, "⌛ Preparing data...")
        try:
            corpus = self.corpus.group_by_time_period(
                time_period_specifier=self.temporal_key, target_column_name=self.category_name
            )
            self.set_buzy(False, "✔")
        except Exception as ex:
            self.set_buzy(False, f"😮 {str(ex)}")

        return corpus

    def _compile(self, **_) -> Any:  # pylint: disable=arguments-differ
        top_terms: pd.DataFrame = self.transform().get_top_terms(
            category_column=self.category_name, n_top=self.top_count, kind=self.kind
        )
        return top_terms

    def plot(self, *, data: Sequence[pd.DataFrame], temporal_key: str, **_) -> None:  # pylint: disable=unused-argument
        self.set_buzy(True, "⌛ Preparing data...")
        try:
            self.clear()
            self.data = self._compile()
            self.load()
            self.set_buzy(False, "✔")
        except Exception as ex:
            self.set_buzy(False, f"😮 {str(ex)}")

        return self

    def load(self, *_):
        try:
            self.set_buzy(True, "⌛ Loading data...")

            if self._output:
                with pd.option_context('display.precision', 2, 'display.max_columns', 300):
                    pd.options.display.max_columns = 300
                    self._output.clear_output()
                    style_dict = [
                        {
                            'selector': 'thead th',
                            'props': [('position', 'sticky'), ('top', '0'), ('background-color', 'grey')],
                        }
                    ]
                    with self._output:  # pylint: disable=not-context-manager
                        df: pd.DataFrame = (
                            self.data.style.format({'H': "{:.2%}"}).set_table_styles(style_dict).hide_index()
                        )
                        display(HTML(df.style.render()))

            if self._table is not None:
                self._table.clear()
                self._table.load(self.data)

            self.set_buzy(False, "✔")

        except Exception as ex:
            self.set_buzy(False, f"😮 {str(ex)}")

    def clear(self, *_) -> "TopTokensDisplayer":
        if self._table is not None:
            self._table.clear()
        if self._output is not None:
            self._output.clear_output()
        return self

    def download(self, *_):
        with contextlib.suppress(Exception):
            with self._download_output:
                js_download = nu.create_js_download(self.data, index=True)
                if js_download is not None:
                    IPython.display.display(js_download)

    def set_buzy(self, is_buzy: bool = True, message: str = None):

        if message:
            self.alert(message)

        self._keyness.disabled = is_buzy
        self._top_count.disabled = is_buzy
        self._compute.disabled = is_buzy
        self._save.disabled = is_buzy
        self._download.disabled = is_buzy
        self._kind.disabled = is_buzy
        self._temporal_key.disabled = is_buzy

    def layout(self) -> GridBox:
        layout: GridBox = GridBox(
            [
                HBox(
                    [
                        self._placeholder,
                        VBox([HTML("<b>Keyness</b>"), self._keyness]),
                        VBox([HTML("<b>Top count</b>"), self._top_count]),
                        VBox([HTML("<b>Grouping</b>"), self._temporal_key]),
                        VBox([HTML("<b>Kind</b>"), self._kind]),
                        VBox(
                            [
                                self._compute,
                                self._download,
                            ]
                        ),
                        VBox([HTML("📌"), self._alert]),
                        self._download_output,
                    ]
                ),
            ]
            + ([] if self.simple_display else [self._table])
            + ([self._output] if self.simple_display else []),
            layout=Layout(width='auto'),
        )
        return layout

    @property
    def keyness(self) -> KeynessMetric:
        return self._keyness.value

    @property
    def data(self) -> pd.DataFrame:
        return self._compile(corpus=self.corpus)

    @property
    def top_count(self) -> int:
        return self._top_count.value

    @property
    def temporal_key(self) -> str:
        return self._temporal_key.value

    @property
    def kind(self) -> str:
        return self._kind.value

    def alert(self, msg: str):
        self._alert.value = msg

    def warn(self, msg: str):
        self.alert(f"<span style='color=red'>{msg}</span>")


class CoOccurrenceTopTokensDisplayer(TopTokensDisplayer):
    def __init__(self, bundle: Bundle, name: str = "TopTokens"):
        super().__init__(corpus=bundle.corpus, name=name)
        self.bundle = bundle
        self._keyness_source: Dropdown = Dropdown(
            options={
                "Full corpus": KeynessMetricSource.Full,
                "Concept corpus": KeynessMetricSource.Concept,
                "Weighed corpus": KeynessMetricSource.Weighed,
            }
            if bundle.concept_corpus is not None
            else {
                "Full corpus": KeynessMetricSource.Full,
            },
            value=KeynessMetricSource.Weighed if bundle.concept_corpus is not None else KeynessMetricSource.Full,
            layout=Layout(width='auto'),
        )

        self._placeholder.children = [VBox([HTML("<b>Source</b>"), self._keyness_source])]

    def keyness_widget(self) -> Dropdown:
        return Dropdown(
            options={
                "TF": KeynessMetric.TF,
                "TF (norm)": KeynessMetric.TF_normalized,
                "TF-IDF": KeynessMetric.TF_IDF,
                "HAL CWR": KeynessMetric.HAL_cwr,
                "PPMI": KeynessMetric.PPMI,
                "LLR": KeynessMetric.LLR,
                "LLR(Z)": KeynessMetric.LLR_Z,
                "LLR(N)": KeynessMetric.LLR_N,
                "DICE": KeynessMetric.DICE,
            },
            value=KeynessMetric.TF,
            layout=Layout(width='auto'),
        )

    @property
    def keyness_source(self) -> KeynessMetricSource:
        return self._keyness_source.value

    def transform(self) -> VectorizedCorpus:
        self.set_buzy(True, f"⌛ Computing {self.keyness.name}...")
        try:
            corpus: VectorizedCorpus = self.bundle.keyness_transform(
                opts=ComputeKeynessOpts(
                    period_pivot=self.temporal_key,
                    keyness=self.keyness,
                    keyness_source=self.keyness_source,
                    fill_gaps=False,
                    normalize=False,
                    tf_threshold=1,
                    pivot_column_name=self.category_name,
                )
            )
            self.set_buzy(False, "✔")
        except Exception as ex:
            self.set_buzy(False, f"😮 {str(ex)}")

        return corpus

    def set_buzy(self, is_buzy: bool = True, message: str = None):
        super().set_buzy(is_buzy=is_buzy, message=message)
        self._keyness_source.disabled = is_buzy
