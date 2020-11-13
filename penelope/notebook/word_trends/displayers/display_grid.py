import pandas as pd
from ipyaggrid import Grid
from IPython.display import display

from . import data_compilers

NAME = "Grid"

compile = data_compilers.compile_year_token_vector_data  # pylint: disable=redefined-builtin


def setup(container, **kwargs):  # pylint: disable=unused-argument
    pass


def default_column_defs(df):
    column_defs = [
        {
            'headerName': column.title(),
            'field': column,
            # 'rowGroup':False,
            # 'hide':False,
            'cellRenderer': "function(params) { return params.value.toFixed(6); }" if column != 'year' else None,
            # 'type': 'numericColumn'
        }
        for column in df.columns
    ]
    return column_defs


def plot(data, **kwargs):  # pylint: disable=unused-argument

    df = pd.DataFrame(data=data).set_index('year')
    column_defs = default_column_defs(df)
    grid_options = {
        'columnDefs': column_defs,
        'enableSorting': True,
        'enableFilter': True,
        'enableColResize': True,
        'enableRangeSelection': False,
    }

    g = Grid(
        grid_data=df,
        columns_fit='auto',
        export_csv=True,
        export_excel=True,
        export_mode="buttons",
        index=True,
        keep_multiindex=False,
        menu={'buttons': [{'name': 'Export Grid', 'hide': True}]},
        quick_filter=False,
        show_toggle_delete=False,
        show_toggle_edit=False,
        theme='ag-theme-balham',
        grid_options=grid_options,
    )

    display(g)
