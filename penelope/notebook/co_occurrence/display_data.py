import contextlib
from typing import List

import IPython.display as IPython_display
import pandas as pd
from ipywidgets import HTML, Button, Dropdown, GridBox, HBox, Layout, Text, ToggleButton, VBox
from pandas.api.types import is_numeric_dtype
from penelope.co_occurrence.convert import store_co_occurrences
from penelope.notebook.utility import create_js_download
from penelope.utility import dotget, path_add_timestamp
from perspective import PerspectiveWidget


def prepare_data_for_display(
    co_occurrences: pd.DataFrame,
    threshold: int = 25,
    match_tokens: List[str] = None,
    skip_tokens: List[str] = None,
    n_head: int = 100000,
) -> pd.DataFrame:

    if len(co_occurrences) > n_head:
        print(f"warning: only {n_head} records out of {len(co_occurrences)} records are displayed.")

    for token in match_tokens or []:
        co_occurrences = co_occurrences[
            (co_occurrences.w1 == token)
            | (co_occurrences.w2 == token)
            | co_occurrences.w1.str.startswith(f"{token}@")
            | co_occurrences.w2.str.startswith(f"{token}@")
        ]

    for token in skip_tokens or []:
        co_occurrences = co_occurrences[
            (co_occurrences.w1 != token)
            & (co_occurrences.w2 != token)
            & ~co_occurrences.w1.str.startswith(f"{token}@")
            & ~co_occurrences.w2.str.startswith(f"{token}@")
        ]

    co_occurrences = co_occurrences.copy()

    co_occurrences["tokens"] = co_occurrences.w1 + "/" + co_occurrences.w2

    global_tokens_counts: pd.Series = co_occurrences.groupby(['tokens'])['value'].sum()
    threshold_tokens: pd.Index = global_tokens_counts[global_tokens_counts >= threshold].index

    co_occurrences = co_occurrences.set_index('tokens').loc[threshold_tokens]  # [['year', 'value', 'value_n_t']]

    if is_numeric_dtype(co_occurrences['value_n_t'].dtype):
        co_occurrences['value_n_t'] = co_occurrences.value_n_t.apply(lambda x: f'{x:.8f}')

    return co_occurrences.head(n_head)


class CoOccurrenceTable(GridBox):  # pylint: disable=too-many-ancestors
    def __init__(
        self,
        co_occurrences: pd.DataFrame,
        *,
        default_token_filter: str = None,
        hide_concept_default: bool = True,
        compute_options: dict = None,
        **kwargs,
    ):
        self.compute_options: dict = compute_options or {}
        self.hide_concept_default: bool = hide_concept_default

        if isinstance(co_occurrences, dict):
            self.co_occurrences: pd.DataFrame = pd.DataFrame(co_occurrences)
        elif isinstance(co_occurrences, pd.DataFrame):
            self.co_occurrences: pd.DataFrame = co_occurrences
        else:
            raise ValueError(f"Data must be dict or pandas.DataFrame not {type(co_occurrences)}")

        # self._output: Output = Output(Layout=Layout(width='auto'))
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
        # Add option for filtering out concept words
        # if len(dotget(self.compute_options, "context_opts.concept", [])) > 0:
        self._show_concept = ToggleButton(description='Show concept', value=False, icon='', layout=Layout(width='auto'))
        self._show_concept.observe(self.update_data, 'value')
        self._show_concept.observe(self.toggle_icon, 'value')
        self._message: HTML = HTML()
        self._save = Button(description='Save data', layout=Layout(width='auto'))
        self._download = Button(description='Download data', layout=Layout(width='auto'))

        #        with contextlib.suppress(Exception):

        self._table: PerspectiveWidget = PerspectiveWidget(
            self.get_data(),
            sort=[["tokens", "asc"]],
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
        data = prepare_data_for_display(
            self.co_occurrences,
            threshold=self._global_threshold_filter.value,
            skip_tokens=self.skip_concept_tokens(),
            match_tokens=self._token_filter.value.strip().split(),
            n_head=self._record_count_limit.value,
        )
        return data

    def toggle_icon(self, event: dict) -> None:
        with contextlib.suppress(Exception):
            event['owner'].icon = 'check' if event['new'] else ''

    def skip_concept_tokens(self) -> List[str]:

        if self._show_concept.value:
            return []

        concept_tokens = dotget(self.compute_options, "context_opts.concept", []) or []

        return concept_tokens

    def update_data(self, *_):
        data = self.get_data()
        self.info(f"Data size: {len(data)}")
        self._table.load(data)

    # def filter(self, _):
    #     # BOOLEAN_FILTERS = ["&", "|", "==", "!=", "or", "and"]
    #     # NUMBER_FILTERS = ["<", ">", "==", "<=", ">=", "!=", "is null", "is not null"]
    #     # STRING_FILTERS = ["==", "contains", "!=", "in", "not in", "begins with", "ends with"]
    #     # DATETIME_FILTERS = ["<", ">", "==", "<=", ">=", "!="]
    #     self._button_bar.disabled = True
    #     filters = []  # self._get_filters()
    #     # with contextlib.suppress(Exception):
    #     #     if self._table.filters != filters:
    #     #         self._table.filters = filters
    #     self.update_data({})
    #     self._button_bar.disabled = False

    # def _get_filters(self) -> List[List[str]]:

    #     filters = []
    #     for token in self._token_filter.value.strip().split():
    #         token.startswith("~")
    #         filters.append(["tokens", "contains", self._token_filter.value])

    #     if self._global_threshold_filter.value > 1:
    #         filters.append(["value", ">=", self._global_threshold_filter.value])

    #     return filters

    def save(self, _b):
        store_co_occurrences(path_add_timestamp('co_occurrence_data.csv'), self.get_data())

    def download(self, *_):
        self._button_bar.disabled = True
        # with self._output:
        with contextlib.suppress(Exception):
            js_download = create_js_download(self.get_data(), index=True)
            if js_download is not None:
                IPython_display.display(js_download)
        self._button_bar.disabled = False
