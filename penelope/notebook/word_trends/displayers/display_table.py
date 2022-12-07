import IPython.display as ip
import ipywidgets as w
import pandas as pd

from penelope.corpus.dtm import WORD_PAIR_DELIMITER
from penelope.notebook import grid_utility as gu
from penelope.notebook.utility import create_js_download
from penelope.utility import try_split_column

# from ...ipyaggrid_utility import display_grid
from .interface import ITrendDisplayer


class TableDisplayer(ITrendDisplayer):
    """Displays data as a pivot table with category as rows and tokens as columns"""

    def __init__(self, name: str = "Table", **opts):
        super().__init__(name=name, **opts)
        self._download_button: w.Button = w.Button(description="Download", layout=dict(width='100px'))
        self._download_button.on_click(self.download)
        self.data: pd.DataFrame = None

    def setup(self, *_, **__):
        return

    def create_data_frame(self, plot_data: dict, category_name: str) -> pd.DataFrame:
        df: pd.DataFrame = pd.DataFrame(data=plot_data)
        df = df[[category_name] + [x for x in df.columns if x != category_name]]
        return df

    def plot(self, *, data: list[pd.DataFrame], temporal_key: str, **_) -> None:  # pylint: disable=unused-argument

        with self.output:
            ip.display(self._download_button)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                widget: gu.TableWidget = gu.table_widget(data[-1])
                ip.display(widget)
            self.data = data[-1]

    def download(self, *_):
        with self.output:
            js_download: ip.Javascript = create_js_download(self.data, index=True)
            if js_download is not None:
                ip.display(js_download)


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
