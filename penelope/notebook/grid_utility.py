from __future__ import annotations

from typing import Callable, Union

import ipydatagrid as dg
import ipywidgets as w
import pandas as pd
from ipydatagrid import DataGrid, TextRenderer
from IPython.display import display

TableWidget = dg.DataGrid


def display_grid(data: Union[dict, pd.DataFrame], **opts) -> DataGrid:
    column_formats: dict = opts.get('column_formats', {})
    renderers = {
        c: TextRenderer(format=column_formats[c]) if c in column_formats else TextRenderer() for c in data.columns
    }
    grid = DataGrid(
        data,
        base_row_size=30,
        selection_mode="cell",
        editable=False,
        base_column_size=80,
        # layout={'height': '200px'}
        renderers=renderers,
    )
    grid.transform([{"type": "sort", "columnIndex": 0, "desc": False}])
    grid.auto_fit_params = {"area": "all", "padding": 40, "numCols": 1}
    grid.auto_fit_columns = True
    return grid


def table_widget(data: pd.DataFrame, **kwargs) -> dg.DataGrid:

    """If handler is passed, then create wrapper handler that passes row as argument"""
    handler: Callable[[pd.Series], None] = kwargs.pop('handler', None)

    _defaults: dict = dict(
        selection_mode="row",
        auto_fit_columns=True,
        auto_fit_params={"area": "body"},
        grid_style={'background_color': '#f9f9f9', 'grid_line_color': '#f9f9f9'},
        # header_visibility='column',
        editable=False,
    )

    g: dg.DataGrid = dg.DataGrid(dataframe=data, **{**_defaults, **kwargs})

    if handler is not None:

        def row_clicked(args) -> None:
            data_id: int = args.get('primary_key_row', None)
            if data_id is not None:
                item: pd.Series = g.data.iloc[data_id]
                handler(item, g)

        g.on_cell_click(row_clicked)

    return g


class DataGridOutput(w.Output):
    def __init__(self):
        super().__init__()
        self.widget: TableWidget = None
        self.data: pd.DataFrame = None

    def update(self, data: pd.DataFrame) -> None:
        self.clear()
        self.data = data
        if self.data is None:
            return
        self.widget = table_widget(self.data)
        with self:
            display(self.widget)

    def load(self, data: pd.DataFrame) -> None:
        self.update(data)

    def clear(self) -> None:
        self.widget = None
        self.clear_output()
