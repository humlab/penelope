from dataclasses import dataclass, field

import pandas as pd
from ipyaggrid import Grid
from IPython.display import display

from ._compile_mixins import CategoryDataMixin
from ._displayer import ITrendDisplayer


@dataclass
class GridDisplayer(CategoryDataMixin, ITrendDisplayer):

    name: str = field(default="Grid")

    def setup(self):
        return

    def default_column_defs(self, df: pd.DataFrame):
        column_defs = [
            {
                'headerName': column.title(),
                'field': column,
                # 'rowGroup':False,
                # 'hide':False,
                'cellRenderer': (
                    "function(params) { return params.value.toFixed(6); }"
                    if column not in ('year', 'category')
                    else None
                ),
                # 'type': 'numericColumn'
            }
            for column in df.columns
        ]
        return column_defs

    def plot(self, plot_data: dict, **_):  # pylint: disable=unused-argument

        df = pd.DataFrame(data=plot_data).set_index('category')

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
