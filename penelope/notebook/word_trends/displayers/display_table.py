from dataclasses import dataclass, field

import IPython.display
import pandas as pd
from penelope.notebook.ipyaggrid_utility import display_grid

from ._compile_mixins import CategoryDataMixin
from ._displayer import ITrendDisplayer


@dataclass
class TableDisplayer(CategoryDataMixin, ITrendDisplayer):

    name: str = field(default="Table")

    def setup(self, *_, **__):
        return

    def plot(self, plot_data: dict, **_):  # pylint: disable=unused-argument

        with self.output:
            df = pd.DataFrame(data=plot_data)
            df = df[['category'] + [x for x in df.columns if x != 'category']]
            g = display_grid(df)
            IPython.display.display(g)
