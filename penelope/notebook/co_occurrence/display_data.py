import contextlib
from typing import List
import IPython.display as IPython_display
import pandas as pd
from ipywidgets import Button, HBox, Layout, Text, ValueWidget, VBox, Dropdown, Output
from penelope.co_occurrence.convert import store_co_occurrences
from penelope.notebook.utility import create_js_download
from penelope.utility import path_add_timestamp

# from ipyregulartable import RegularTableWidget
from perspective import PerspectiveWidget


class CoOccurrenceTable(VBox, ValueWidget):
    def __init__(self, data: pd.DataFrame, default_token_filter: str = None, default_count_filte: int=25, **kwargs):

        if isinstance(data, dict):
            self._data: pd.DataFrame = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            self._data: pd.DataFrame = data
        else:
            raise ValueError(f"Data must be dict or pandas.DataFrame not {type(data)}")

        self._data["tokens"] = data.w1 + "/" + data.w2
        #self._data.drop(["w1", "w2"])
        self._data: pd.DataFrame = pd.DataFrame(data=data)  # .set_index('year')
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
            value=default_count_filte,
            placeholder='value filter',
            layout=Layout(width='auto'),
        )
        self._filter = Button(description='Filter', layout=Layout(width='auto'))
        self._save = Button(description='Save', layout=Layout(width='auto'))
        self._download = Button(description='Download', layout=Layout(width='auto'))
        # self._table: RegularTableWidget = RegularTableWidget(data)
        self._table: PerspectiveWidget = PerspectiveWidget(data, filters=self._get_filters())
        self._button_bar = HBox(
            children=[self._token_filter, self._value_filter, self._filter, self._save, self._download, self._output],
            layout=Layout(width='auto'),
        )

        super().__init__(children=[self._button_bar, self._table], layout=Layout(width='auto'), **kwargs)

        # self._token_filter.observe(self.filter, names='value')
        # self._value_filter.observe(self.filter, names='value')

        self._filter.on_click(self.filter)
        self._save.on_click(self.save)
        self._download.on_click(self.download)

    def filter(self, _):
        # BOOLEAN_FILTERS = ["&", "|", "==", "!=", "or", "and"]
        # NUMBER_FILTERS = ["<", ">", "==", "<=", ">=", "!=", "is null", "is not null"]
        # STRING_FILTERS = ["==", "contains", "!=", "in", "not in", "begins with", "ends with"]
        # DATETIME_FILTERS = ["<", ">", "==", "<=", ">=", "!="]
        filters = self._get_filters()
        if self._table.filters != filters:
            self._table.filters = filters

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
