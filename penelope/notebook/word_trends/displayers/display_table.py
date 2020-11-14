import IPython.display
import pandas as pd
from penelope.notebook.ipyaggrid_utility import display_grid

from ._displayer import ITrendDisplayer, YearTokenDataMixin


class TableDisplayer(ITrendDisplayer, YearTokenDataMixin):

    name = "Table"

    def setup(self, *_, **__):
        pass

    def plot(self, data, **_):  # pylint: disable=unused-argument

        with self.output:
            df = pd.DataFrame(data=data)
            df = df[['year'] + [x for x in df.columns if x != 'year']].set_index('year')
            g = display_grid(df)
            IPython.display.display(g)
