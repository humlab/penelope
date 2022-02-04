from __future__ import annotations

from typing import Callable, Union

import ipydatagrid as dg
import pandas as pd
from ipydatagrid import DataGrid, TextRenderer

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


def table_widget(data: pd.DataFrame, **kwargs) -> None:

    """If handler is passed, then create wrapper handler that passes row as argument"""
    handler: Callable[[pd.Series], None] = kwargs.pop('handler', None)

    _defaults: dict = dict(
        selection_mode="row",
        auto_fit_columns=True,
        auto_fit_params={"area": "body"},
        grid_style={'background_color': '#dcdcdc', 'grid_line_color': '#dcdcdc'},
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
