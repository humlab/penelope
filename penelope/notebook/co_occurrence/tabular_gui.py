import contextlib
import fnmatch
from collections.abc import Iterable
from typing import List, Set

import IPython.display as IPython_display
import numpy as np
import pandas as pd
from ipywidgets import HTML, Button, Dropdown, GridBox, HBox, Layout, Output, Text, ToggleButton, VBox
from penelope.co_occurrence import Bundle, CoOccurrenceHelper, store_co_occurrences
from penelope.common.keyness import KeynessMetric
from penelope.corpus import Token2Id, VectorizedCorpus
from penelope.notebook.utility import create_js_download
from penelope.utility import path_add_timestamp
from perspective import PerspectiveWidget

# pylint: disable=too-many-instance-attributes
DISPLAY_COLUMNS = ['time_period', 'w1', 'w2', 'value']


def empty_data():
    return pd.DataFrame(
        data={
            'time_period': pd.Series(data=[], dtype=np.int),
            'w1': pd.Series(data=[], dtype=str),
            'w2': pd.Series(data=[], dtype=str),
            'value': pd.Series(data=[], dtype=str),
        }
    )


CURRENT_BUNDLE = None

# from IPython.display import display
# import panel as pn
# from bokeh.models.widgets.tables import NumberFormatter, StringFormatter
# PANEL_FORMATTERS = {
#     'category': NumberFormatter(format='0'),
#     'time_period': NumberFormatter(format='0'),
#     'year': NumberFormatter(format='0'),
#     'index': StringFormatter(),
# }

# class TabulatorTableView:
#     def __init__(self, data=None):  # pylint: disable=unused-argument
#         self.container = Output()
#         self.table: pn.widgets.Tabulator = None

#     def update(self, data):
#         if self.table is None:
#             self.table: pn.widgets.Tabulator = pn.widgets.Tabulator(
#                 value=data,
#                 formatters=PANEL_FORMATTERS,
#                 layout='fit_data_table',
#                 # pagination='remote',
#                 # hidden_columns=['index'],
#                 row_height=24,
#                 show_index=False,
#             )
#             self.table.auto_edit = False
#             self.container.clear_output()
#             with self.container:
#                 display(self.table)
#         else:
#             self.table.value = data


class PerspectiveTableView:
    def __init__(self, data: pd.DataFrame = None):
        data = data if data is not None else empty_data()
        self.table: PerspectiveWidget = PerspectiveWidget(
            data,
            sort=[["token", "asc"]],
            aggregates={},
        )
        self.container = self.table
        self.precision: int = 6

    def update(self, data: pd.DataFrame) -> None:
        self.table.replace(
            data={
                'time_period': data.time_period.apply(str),
                'w1': data.w1,
                'w2': data.w2,
                'value': (
                    data.value.apply(str)
                    if np.issubdtype(data.value.dtype, np.integer)
                    else data.value.apply(f"{{:10.{self.precision}f}}".format)
                ),
            }
        )

    # def format_columns(self, data: pd.DataFrame, precision: int = 6):
    #     data = pd.DataFrame(data=data)
    #     for column in data.columns:
    #         if np.issubdtype(data[column].dtype, np.integer):
    #             data[column] = data[column].apply(str)
    #         if np.issubdtype(data[column].dtype, np.inexact):
    #             data[column] = data[column].apply(f"{{:10.{precision}f}}".format)
    #     return data


class TabularCoOccurrenceGUI(GridBox):  # pylint: disable=too-many-ancestors
    def __init__(
        self,
        *,
        bundle: Bundle,
        default_token_filter: str = None,
        **kwargs,
    ):
        global CURRENT_BUNDLE
        CURRENT_BUNDLE = bundle

        """Alternative implementation that uses VectorizedCorpus"""
        self.bundle = bundle
        self.co_occurrences: pd.DataFrame = None
        self.pivot_column_name: str = 'time_period'

        if not isinstance(bundle.token2id, Token2Id):
            raise ValueError(f"Expected Token2Id, found {type(bundle.token2id)}")

        if not isinstance(bundle.compute_options, dict):
            raise ValueError("Expected Compute Options in bundle but found no such thing.")

        """Current processed corpus"""
        self.corpus = bundle.corpus

        """Properties that changes current corpus"""
        self._pivot: Dropdown = Dropdown(
            options=["year", "lustrum", "decade"],
            value="decade",
            placeholder='Group by',
            layout=Layout(width='auto'),
        )

        """Properties that changes current corpus"""
        self._keyness: Dropdown = Dropdown(
            options={
                "TF": KeynessMetric.TF,
                "TF (norm)": KeynessMetric.TF_normalized,
                "TF-IDF": KeynessMetric.TF_IDF,
                "HAL CWR": KeynessMetric.HAL_cwr,
                "PPMI": KeynessMetric.PPMI,
                "LLR": KeynessMetric.LLR,
                "LLR(D)": KeynessMetric.LLR_Dunning,
                "DICE": KeynessMetric.DICE,
            },
            value=KeynessMetric.TF,
            layout=Layout(width='auto'),
        )

        """Properties that don't change current corpus"""
        # self._rank: Dropdown = Dropdown(
        #     options=[10 ** i for i in range(0, 7)],
        #     value=10000,
        #     layout=Layout(width='auto'),
        # )
        self._token_filter: Text = Text(
            value=default_token_filter, placeholder='token match', layout=Layout(width='auto')
        )
        self._global_threshold_filter: Dropdown = Dropdown(
            options={f'>= {i}': i for i in (1, 2, 3, 4, 5, 10, 25, 50, 100, 250, 500)},
            value=5,
            layout=Layout(width='auto'),
        )
        self.concepts: Set[str] = set(self.bundle.context_opts.concept or [])
        self._largest: Dropdown = Dropdown(
            options=[10 ** i for i in range(0, 7)],
            value=10000,
            layout=Layout(width='auto'),
        )
        self._show_concept = ToggleButton(description='Show concept', value=False, icon='', layout=Layout(width='auto'))
        self._message: HTML = HTML()
        self._save = Button(description='Save', layout=Layout(width='auto'))
        # self._display = Button(description='Update', layout=Layout(width='auto'))
        self._download = Button(description='Download', layout=Layout(width='auto'))
        self._download_output: Output = Output()
        self._table_view = PerspectiveTableView(data=empty_data())

        # self._toggle2 = ToggleButton(description='Use Load', value=True, icon='', layout=Layout(width='auto'))
        # self._toggle2 = ToggleButton(description='🔨', value=True, icon='', layout=Layout(width='auto'))

        self._button_bar = HBox(
            children=[
                VBox([HTML("<b>Token match</b>"), self._token_filter]),
                VBox([HTML("<b>Keyness metric</b>"), self._keyness]),
                VBox([HTML("🙂"), self._show_concept if self._show_concept is not None else HTML("")]),
                VBox([HTML("<b>Group by</b>"), self._pivot]),
                VBox([HTML("<b>Global threshold</b>"), self._global_threshold_filter]),
                VBox([HTML("<b>Group limit</b>"), self._largest]),
                # VBox([HTML("<b>Group ranks</b>"), self._rank]),
                # VBox([HTML("<b>Result limit</b>"), self._head]),
                VBox([self._save, self._download]),
                # VBox([self._toggle2, self._toggle2]),
                # VBox([self._display]),
                VBox([HTML("📌"), self._message]),
                self._download_output,
            ],
            layout=Layout(width='auto'),
        )
        super().__init__(children=[self._button_bar, self._table_view.container], layout=Layout(width='auto'), **kwargs)

        self._save.on_click(self.save)
        # self._display.on_click(self.save)
        self._download.on_click(self.download)

        self.start_observe()

    def set_buzy(self, is_buzy: bool = True, message: str = None):

        if message:
            self.alert(message)

        self._keyness.disabled = is_buzy
        self._show_concept.disabled = is_buzy
        self._pivot.disabled = is_buzy
        self._global_threshold_filter.disabled = is_buzy
        self._token_filter.disabled = is_buzy
        self._save.disabled = is_buzy
        self._download.disabled = is_buzy
        self._largest.disabled = is_buzy

    def start_observe(self):

        self.stop_observe()

        self._keyness.observe(self._update_corpus, 'value')

        self._show_concept.observe(self._update_co_occurrences, 'value')
        self._show_concept.observe(self._update_toggle_icon, 'value')

        self._pivot.observe(self._update_corpus, 'value')
        self._global_threshold_filter.observe(self._update_corpus, 'value')

        # self._head.observe(self._update_co_occurrences, 'value')
        self._token_filter.observe(self._filter_co_occurrences, 'value')
        self._largest.observe(self._update_co_occurrences, 'value')

        return self

    def stop_observe(self):

        with contextlib.suppress(Exception):

            self._keyness.unobserve(self._update_corpus, 'value')

            self._show_concept.unobserve(self._update_co_occurrences, 'value')
            self._show_concept.unobserve(self._update_toggle_icon, 'value')

            self._pivot.unobserve(self._update_corpus, 'value')
            self._global_threshold_filter.unobserve(self._update_corpus, 'value')

            # self._head.unobserve(self._update_co_occurrences, 'value')
            self._token_filter.unobserve(self._filter_co_occurrences, 'value')
            self._largest.unobserve(self._update_co_occurrences, 'value')

    def alert(self, message: str) -> None:
        self._message.value = f"<span style='color: red; font-weight: bold;'>{message}</span>"

    def info(self, message: str) -> None:
        self._message.value = f"<span style='color: green; font-weight: bold;'>{message}</span>"

    def _update_corpus(self, *_):

        self.corpus = self.to_corpus()

        self._update_co_occurrences()

    def _update_co_occurrences(self, *_) -> pd.DataFrame:

        self.co_occurrences = self.to_co_occurrences()

        # with contextlib.suppress(PerspectiveError, TraitError):

        self.set_buzy(True, "⌛ loading table...")

        self._table_view.update(self.co_occurrences[DISPLAY_COLUMNS])

        self.set_buzy(False)

        self.info(f"Data size: {len(self.co_occurrences)}")

    def _filter_co_occurrences(self, *_) -> pd.DataFrame:

        # with contextlib.suppress(PerspectiveError, TraitError):
        co_occurrences: pd.DataFrame = self.to_filtered_co_occurrences()

        self._table_view.update(co_occurrences[DISPLAY_COLUMNS])

        self.info(f"Data size: {len(co_occurrences)}")

    def _update_toggle_icon(self, event: dict) -> None:
        with contextlib.suppress(Exception):
            event['owner'].icon = 'check' if event['new'] else ''

    @property
    def ignores(self) -> List[str]:

        if self.show_concept or not self.concepts:
            return set()

        if isinstance(self.concepts, Iterable):
            return set(self.concepts)

        return {self.concepts}

    @property
    def show_concept(self) -> bool:
        return self._show_concept.value

    @show_concept.setter
    def show_concept(self, value: bool):
        self._show_concept.value = value

    @property
    def keyness(self) -> KeynessMetric:
        return self._keyness.value

    @keyness.setter
    def keyness(self, value: KeynessMetric):
        self._keyness.value = value

    @property
    def global_threshold(self) -> int:
        return self._global_threshold_filter.value

    @global_threshold.setter
    def global_threshold(self, value: int):
        self._global_threshold_filter.value = value

    @property
    def largest(self) -> int:
        return self._largest.value

    @largest.setter
    def largest(self, value: int):
        self._largest.value = value

    @property
    def token_filter(self) -> List[str]:
        return self._token_filter.value.strip().split()

    @token_filter.setter
    def token_filter(self, value: List[str]):
        self._token_filter.value = ' '.join(value) if isinstance(value, list) else value

    @property
    def pivot(self) -> str:
        return self._pivot.value

    @pivot.setter
    def pivot(self, value: str):
        self._pivot.value = value

    def save(self, *_b):
        store_co_occurrences(
            filename=path_add_timestamp('co_occurrence_data.csv'),
            co_occurrences=self.co_occurrences,
            store_feather=False,
        )

    def download(self, *_):
        self._button_bar.disabled = True

        with contextlib.suppress(Exception):
            js_download = create_js_download(self.co_occurrences, index=True)
            if js_download is not None:
                with self._download_output:
                    IPython_display.display(js_download)

        self._button_bar.disabled = False

    def to_co_occurrences(self) -> pd.DataFrame:

        self.set_buzy(True, "⌛ preparing co-occurrences...")

        try:

            if self.pivot_column_name not in self.corpus.document_index.columns:
                raise ValueError(
                    f"expected '{self.pivot_column_name}' but not found in {', '.join(self.corpus.document_index.columns)}"
                )

            co_occurrences: pd.DataFrame = (
                CoOccurrenceHelper(
                    corpus=self.corpus,
                    source_token2id=self.bundle.token2id,
                    pivot_keys=self.pivot_column_name,
                )
                .exclude(self.ignores)
                .largest(self.largest)
            ).value

            self.set_buzy(False, None)
            self.alert("✔")
        except Exception as ex:
            self.set_buzy(False)
            self.alert(f"😢 {str(ex)}")
            raise

        return co_occurrences

    def to_filtered_co_occurrences(self) -> pd.DataFrame:

        if not self.token_filter:
            return self.co_occurrences

        co_occurrences: pd.DataFrame = self.co_occurrences

        re_filters: List[str] = [fnmatch.translate(s) for s in self.token_filter]

        for re_filter in re_filters:
            co_occurrences = co_occurrences[
                co_occurrences.token.astype(str).str.contains(pat=re_filter, case=False, na="")
            ]

        return co_occurrences

    def to_corpus(self) -> VectorizedCorpus:
        """Returns a grouped, optionally TF-IDF, corpus filtered by token & threshold."""
        self.set_buzy(True, "⌛ updating corpus...")

        # print(f"to_corpus: keyness={self.keyness}")
        # print(f"to_corpus: global_threshold={self.global_threshold}")
        # print(f"to_corpus: pivot={self.pivot}")
        # print(f"to_corpus: pivot_column_name={self.pivot_column_name}")
        # print(f"to_corpus: corpus.shape (pre)={self.bundle.corpus.data.shape}")

        try:
            corpus: VectorizedCorpus = self.bundle.to_keyness_corpus(
                period_pivot=self.pivot,
                keyness=self.keyness,
                global_threshold=self.global_threshold,
                pivot_column_name=self.pivot_column_name,
                normalize=False,
            )
            # print(f"to_corpus: corpus.shape (post)={corpus.data.shape}")
            self.set_buzy(False, None)
            self.alert("✔")
        except Exception as ex:
            self.set_buzy(False)
            self.alert(f"😢 {str(ex)}")
            raise

        return corpus

    def setup(self) -> "TabularCoOccurrenceGUI":
        self._update_corpus()
        return self
