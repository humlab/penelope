from dataclasses import dataclass, field

import IPython.display
import pandas as pd
from penelope.notebook.ipyaggrid_utility import display_grid
from penelope.utility import try_split_column

from ._compile_mixins import CategoryDataMixin
from ._displayer import ITrendDisplayer


@dataclass
class TableDisplayer(CategoryDataMixin, ITrendDisplayer):
    """Displays data as a pivot table with category as rows and tokens as columns"""

    name: str = field(default="Table")

    def setup(self, *_, **__):
        return

    def create_data_frame(self, plot_data: dict) -> pd.DataFrame:
        df = pd.DataFrame(data=plot_data)
        df = df[['category'] + [x for x in df.columns if x != 'category']]
        return df

    def plot(self, plot_data: dict, **_):  # pylint: disable=unused-argument

        with self.output:
            df = self.create_data_frame(plot_data)
            g = display_grid(df)
            IPython.display.display(g)


@dataclass
class UnnestedTableDisplayer(TableDisplayer):
    """Unnests (unpivots) the pivot table so that tokens columns are turned rows with token category & token count columns"""

    name: str = field(default="Data")

    def create_data_frame(self, plot_data: dict) -> pd.DataFrame:
        df = super().create_data_frame(plot_data)
        df = df.melt(id_vars=["category"], var_name="token", value_name="count")
        return df


@dataclass
class UnnestedExplodeTableDisplayer(UnnestedTableDisplayer):
    """Probes the token column and explodes it to multiple columns if it contains token-pairs and/or PoS-tags"""

    name: str = field(default="Data")

    def create_data_frame(self, plot_data: dict) -> pd.DataFrame:
        df: pd.DataFrame = super().create_data_frame(plot_data)

        df = try_split_column(df, "token", "/", ["w1", "w2"], drop_source=False)

        if 'w1' not in df.columns:
            df = try_split_column(df, "token", "@", ["token", "pos"], drop_source=False)
        else:
            df = try_split_column(df, "w1", "@", ["w1", "w1_pos"], drop_source=False)
            df = try_split_column(df, "w2", "@", ["w2", "w2_pos"], drop_source=False)

        return df
