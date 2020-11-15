from dataclasses import dataclass, field

import IPython.display
import pandas as pd
from penelope.notebook.ipyaggrid_utility import display_grid

from ._displayer import ITrendDisplayer, YearTokenDataMixin


@dataclass
class TableDisplayer(YearTokenDataMixin, ITrendDisplayer):

    name: str = field(default="Table")

    def setup(self, *_, **__):
        return

    def plot(self, data, **_):

        with self.output:
            df = pd.DataFrame(data=data)
            df = df[['year'] + [x for x in df.columns if x != 'year']].set_index('year')
            g = display_grid(df)
            IPython.display.display(g)
