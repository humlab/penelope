import contextlib
from typing import List

import IPython.display as IPython_display
import pandas as pd
from ipywidgets import Button, Dropdown, HBox, Layout, Output, Text, ValueWidget, VBox
from penelope.co_occurrence.convert import store_co_occurrences
from penelope.notebook.utility import create_js_download
from penelope.utility import path_add_timestamp
from perspective import PerspectiveWidget, Sort


def truncate_to_threshold(co_occurrences: pd.DataFrame, threshold: int = 25) -> pd.DataFrame:

    co_occurrences["tokens"] = co_occurrences.w1 + "/" + co_occurrences.w2

    global_tokens_counts: pd.Series = co_occurrences.groupby(['tokens'])['value'].sum()
    threshold_tokens: pd.Index = global_tokens_counts[global_tokens_counts >= threshold].index

    co_occurrences = co_occurrences.set_index('tokens').loc[threshold_tokens][['year', 'value', 'value_n_t']]
    co_occurrences['co_occurrences'] = co_occurrences.value_n_t.apply(lambda x: f'{x:.8f}')
    return co_occurrences


class CoOccurrenceTable(VBox, ValueWidget):  # pylint: disable=too-many-ancestors
    def __init__(
        self,
        co_occurrences: pd.DataFrame,
        global_tokens_count_threshold: int = 25,
        default_token_filter: str = None,
        default_count_filter: int = 25,
        **kwargs,
    ):

        if isinstance(co_occurrences, dict):
            self._data: pd.DataFrame = pd.DataFrame(co_occurrences)
        elif isinstance(co_occurrences, pd.DataFrame):
            self._data: pd.DataFrame = co_occurrences
        else:
            raise ValueError(f"Data must be dict or pandas.DataFrame not {type(co_occurrences)}")

        self._data = truncate_to_threshold(co_occurrences, threshold=global_tokens_count_threshold)

        self._output: Output = Output(Layout=Layout(width='auto'))
        self._token_filter = Text(value=default_token_filter, placeholder='token filter', layout=Layout(width='auto'))
        self._value_filter = Dropdown(
            options={
                '(no filter)': 1,
                'value >= 2': 2,
                'value >= 3': 3,
                'value >= 4': 4,
                'value >= 5': 5,
                'value >= 6': 6,
                'value >= 7': 7,
                'value >= 8': 8,
                'value >= 9': 9,
                'value >= 10': 10,
                'value >= 25': 25,
                'value >= 100': 100,
            },
            value=default_count_filter,
            placeholder='value filter',
            layout=Layout(width='auto'),
        )
        self._filter = Button(description='Filter', layout=Layout(width='auto'))
        self._save = Button(description='Save', layout=Layout(width='auto'))
        self._download = Button(description='Download', layout=Layout(width='auto'))

        self._table: PerspectiveWidget = PerspectiveWidget(
            self._data, filters=self._get_filters(), aggregates={}, sort=[["index", Sort.ASC]]
        )
        self._button_bar = HBox(
            children=[self._token_filter, self._value_filter, self._filter, self._save, self._download, self._output],
            layout=Layout(width='auto'),
        )

        super().__init__(children=[self._button_bar, self._table], layout=Layout(width='auto'), **kwargs)

        self._filter.on_click(self.filter)
        self._save.on_click(self.save)
        self._download.on_click(self.download)

    def filter(self, _):
        # BOOLEAN_FILTERS = ["&", "|", "==", "!=", "or", "and"]
        # NUMBER_FILTERS = ["<", ">", "==", "<=", ">=", "!=", "is null", "is not null"]
        # STRING_FILTERS = ["==", "contains", "!=", "in", "not in", "begins with", "ends with"]
        # DATETIME_FILTERS = ["<", ">", "==", "<=", ">=", "!="]
        self._button_bar.disabled = True
        filters = self._get_filters()
        with contextlib.suppress(Exception):
            if self._table.filters != filters:
                self._table.filters = filters
        self._button_bar.disabled = False

    def _get_filters(self) -> List[List[str]]:

        filters = []
        if self._token_filter.value.strip():
            filters.append(["tokens", "contains", self._token_filter.value])

        if self._value_filter.value > 1:
            filters.append(["value", ">=", self._value_filter.value])

        return filters

    def save(self, _b):
        store_co_occurrences(path_add_timestamp('co_occurrence_data.csv'), self._data)

    def download(self, *_):
        self._button_bar.disabled = True
        with self._output:
            with contextlib.suppress(Exception):
                js_download = create_js_download(self._data, index=True)
                if js_download is not None:
                    IPython_display.display(js_download)
        self._button_bar.disabled = False
