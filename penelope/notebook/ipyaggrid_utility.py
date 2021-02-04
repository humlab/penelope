from typing import Dict, List, Union

import ipyaggrid
import numpy as np
import pandas as pd

DEFAULT_GRID_STYLE = dict(
    columns_fit="auto",
    export_csv=True,
    export_excel=True,
    export_mode="buttons",
    index=True,
    keep_multiindex=False,
    menu={"buttons": [{"name": "Export Grid", "hide": True}]},
    quick_filter=True,
    show_toggle_delete=False,
    show_toggle_edit=False,
    theme="ag-theme-balham",
)

DEFAULT_GRID_OPTIONS = dict(
    enableSorting=True,
    enableFilter=True,
    enableColResize=True,
    enableRangeSelection=False,
    rowSelection='multiple',
)


def default_column_defs(df: pd.DataFrame, precision: int = 6) -> List[Dict]:
    column_defs = [
        {
            "headerName": x[0],
            "field": x[0],
            "cellRenderer": "function(params) { return params.value.toFixed(" + str(precision) + "); }"
            if np.issubdtype(x[1], np.inexact)
            else None,
        }
        for x in zip(df.columns, df.dtypes)
    ]
    return column_defs


def display_grid(
    data: Union[dict, pd.DataFrame],
    column_defs: List[dict] = None,
    grid_options: dict = None,
    grid_style: dict = None,
):

    if isinstance(data, dict):
        df = pd.DataFrame(data=data)  # .set_index('year')
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError(f"Data must be dict or pandas.DataFrame not {type(data)}")

    # if os.environ.get('VSCODE_LOG_STACK', None) is not None:
    #     logging.warning("bug-check: vscode detected, aborting plot...")
    #     return df

    column_defs = default_column_defs(df) if column_defs is None else column_defs

    grid_options = {
        'columnDefs': column_defs,
        **DEFAULT_GRID_OPTIONS,
        **(grid_options or {}),
    }

    grid_style = {**DEFAULT_GRID_STYLE, **(grid_style or {})}

    g = ipyaggrid.Grid(grid_data=df, grid_options=grid_options, **grid_style)

    return g
