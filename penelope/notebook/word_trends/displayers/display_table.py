from typing import Sequence

import IPython.display
import pandas as pd

from penelope.corpus.dtm import WORD_PAIR_DELIMITER
from penelope.utility import try_split_column

# from ...ipyaggrid_utility import display_grid
from .interface import ITrendDisplayer


class TableDisplayer(ITrendDisplayer):
    """Displays data as a pivot table with category as rows and tokens as columns"""

    def __init__(self, name: str = "Table", **opts):
        super().__init__(name=name, **opts)

    def setup(self, *_, **__):
        return

    def create_data_frame(self, plot_data: dict, category_name: str) -> pd.DataFrame:
        df = pd.DataFrame(data=plot_data)
        df = df[[category_name] + [x for x in df.columns if x != category_name]]
        return df

    def plot(self, *, data: Sequence[pd.DataFrame], temporal_key: str, **_) -> None:  # pylint: disable=unused-argument

        with self.output:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                # g = display_grid(data[-1])
                # IPython.display.display(g)
                IPython.display.display(data[-1])
                print("Note: data is plotted using a simplified table view (development)")
                # datagrid = ipydatagrid.DataGrid(data[-1], selection_mode="cell")
                # IPython.display.display(datagrid)


class UnnestedTableDisplayer(TableDisplayer):
    """Unnests (unpivots) the pivot table so that tokens columns are turned rows with token category & token count columns"""

    def __init__(self, name: str = "Table", **opts):
        super().__init__(name=name, **opts)

    def create_data_frame(self, plot_data: dict, category_name: str) -> pd.DataFrame:
        df = super().create_data_frame(plot_data, category_name)
        df = df.melt(id_vars=[category_name], var_name="token", value_name="count")
        return df


class UnnestedExplodeTableDisplayer(UnnestedTableDisplayer):
    """Probes the token column and explodes it to multiple columns if it contains token-pairs and/or PoS-tags"""

    def __init__(self, name: str = "Tabular", **opts):
        super().__init__(name=name, **opts)

    def create_data_frame(self, plot_data: dict, category_name: str) -> pd.DataFrame:
        df: pd.DataFrame = super().create_data_frame(plot_data, category_name)

        df = try_split_column(df, "token", WORD_PAIR_DELIMITER, ["w1", "w2"], drop_source=False)

        if 'w1' not in df.columns:
            df = try_split_column(df, "token", "@", ["token", "pos"], drop_source=False)
        else:
            df = try_split_column(df, "w1", "@", ["w1", "w1_pos"], drop_source=False)
            df = try_split_column(df, "w2", "@", ["w2", "w2_pos"], drop_source=False)

        return df
