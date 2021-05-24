import contextlib
from collections.abc import Iterable
from typing import List, Set

import IPython.display as IPython_display
import pandas as pd
from ipywidgets import HTML, Button, Dropdown, GridBox, HBox, Layout, Output, Text, ToggleButton, VBox
from penelope.co_occurrence import CoOccurrenceHelper, store_co_occurrences
from penelope.corpus import DocumentIndex, Token2Id
from penelope.notebook.utility import create_js_download
from penelope.utility import path_add_timestamp
from perspective import PerspectiveWidget


# pylint: disable=too-many-instance-attributes
class CoOccurrenceTable(GridBox):  # pylint: disable=too-many-ancestors
    def __init__(
        self,
        *,
        co_occurrences: pd.DataFrame,
        token2id: Token2Id,
        document_index: DocumentIndex,
        concepts: Set[str],
        default_token_filter: str = None,
        **kwargs,
    ):

        if not isinstance(co_occurrences, (dict, pd.DataFrame)):
            raise ValueError(f"Expected dict or DataFrame, found {type(co_occurrences)}")

        if not isinstance(token2id, Token2Id):
            raise ValueError(f"Expected Token2Id, found {type(token2id)}")

        self.co_occurrences: pd.DataFrame = (
            pd.DataFrame(co_occurrences) if isinstance(co_occurrences, dict) else co_occurrences
        )

        self.token2id: pd.DataFrame = token2id
        self.document_index: pd.DataFrame = document_index
        self.concepts: Set[str] = concepts

        self.helper: CoOccurrenceHelper = CoOccurrenceHelper(
            self.co_occurrences,
            self.token2id,
            self.document_index,
        )
        self._token_filter: Text = Text(
            value=default_token_filter, placeholder='token match', layout=Layout(width='auto')
        )
        self._global_threshold_filter: Dropdown = Dropdown(
            options={f'>= {i}': i for i in (1, 2, 3, 4, 5, 10, 25, 50, 100, 250, 500)},
            value=5,
            layout=Layout(width='auto'),
        )
        self._record_count_limit: Dropdown = Dropdown(
            options=[10 ** i for i in range(0, 7)],
            value=10000,
            placeholder='Record count limit',
            layout=Layout(width='auto'),
        )
        self._show_concept = ToggleButton(description='Show concept', value=False, icon='', layout=Layout(width='auto'))
        self._show_concept.observe(self.update_data, 'value')
        self._show_concept.observe(self.toggle_icon, 'value')
        self._message: HTML = HTML()
        self._save = Button(description='Save data', layout=Layout(width='auto'))
        self._download = Button(description='Download data', layout=Layout(width='auto'))
        self._download_output: Output = Output()

        self._table: PerspectiveWidget = PerspectiveWidget(
            self.get_data(),
            sort=[["token", "asc"]],
            aggregates={},
        )

        self._button_bar = HBox(
            children=[
                VBox([HTML("<b>Token match</b>"), self._token_filter]),
                VBox([HTML("<b>Global threshold</b>"), self._global_threshold_filter]),
                VBox([HTML("<b>Output limit</b>"), self._record_count_limit]),
                VBox([HTML("👀"), self._show_concept if self._show_concept is not None else HTML("")]),
                VBox([self._save, self._download]),
                VBox([HTML("😢"), self._message]),
                self._download_output,
            ],
            layout=Layout(width='auto'),
        )
        super().__init__(children=[self._button_bar, self._table], layout=Layout(width='auto'), **kwargs)

        # self._filter.on_click(self.update_data)
        self._save.on_click(self.save)
        self._download.on_click(self.download)
        self._global_threshold_filter.observe(self.update_data, 'value')
        self._record_count_limit.observe(self.update_data, 'value')
        self._token_filter.observe(self.update_data, 'value')

    def alert(self, message: str) -> None:
        self._message.value = f"<span style='color: red; font-weight: bold;'>{message}</span>"

    def info(self, message: str) -> None:
        self._message.value = f"<span style='color: green; font-weight: bold;'>{message}</span>"

    def get_data(self):

        data: pd.DataFrame = (
            self.helper.reset()
            .groupby('year')
            .match(self.token_filter)
            .trunk_by_global_count(self.global_threshold)
            .exclude(self.ignores)
            .head(self.count_limit)
        )

        return data

    def toggle_icon(self, event: dict) -> None:
        with contextlib.suppress(Exception):
            event['owner'].icon = 'check' if event['new'] else ''

    @property
    def ignores(self) -> List[str]:

        if self._show_concept.value or not self.concepts:
            return set()

        if isinstance(self.concepts, Iterable):
            return set(self.concepts)

        return {self.concepts}

    @property
    def global_threshold(self) -> int:
        return self._global_threshold_filter.value

    @property
    def count_limit(self) -> int:
        return self._record_count_limit.value

    @property
    def token_filter(self) -> List[str]:
        return self._token_filter.value.strip().split()

    def update_data(self, *_):
        data = self.get_data()
        self.info(f"Data size: {len(data)}")
        self._table.load(data)

    def save(self, _b):
        store_co_occurrences(
            filename=path_add_timestamp('co_occurrence_data.csv'), co_occurrences=self.get_data(), store_feather=False
        )

    def download(self, *_):
        self._button_bar.disabled = True

        with contextlib.suppress(Exception):
            js_download = create_js_download(self.get_data(), index=True)
            if js_download is not None:
                with self._download_output:
                    IPython_display.display(js_download)

        self._button_bar.disabled = False
