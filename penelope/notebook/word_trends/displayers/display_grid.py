from dataclasses import dataclass, field

import pandas as pd
from ipyaggrid import Grid
from IPython.display import display

from ._displayer import ITrendDisplayer, YearTokenDataMixin


@dataclass
class GridDisplayer(YearTokenDataMixin, ITrendDisplayer):

    name: str = field(default="Grid")

    def setup(self):
        return

    def default_column_defs(self, df):
        column_defs = [
            {
                'headerName': column.title(),
                'field': column,
                # 'rowGroup':False,
                # 'hide':False,
                'cellRenderer': ("function(params) { return params.value.toFixed(6); }" if column != 'year' else None),
                # 'type': 'numericColumn'
            }
            for column in df.columns
        ]
        return column_defs

    def plot(self, data, **_):

        df = pd.DataFrame(data=data).set_index('year')
        column_defs = self.default_column_defs(df)
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

        with self.output:
            display(g)
