from __future__ import annotations

from typing import Callable

import ipydatagrid as dg
import pandas as pd

TableWidget = dg.DataGrid


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
