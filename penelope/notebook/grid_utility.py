from typing import Union

import pandas as pd
from ipydatagrid import DataGrid, TextRenderer


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
