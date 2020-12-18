from dataclasses import dataclass, field

import IPython.display
import pandas as pd
from penelope.corpus import VectorizedCorpus
from penelope.notebook.ipyaggrid_utility import display_grid

from ._displayer import ITrendDisplayer, YearTokenDataMixin


@dataclass
class TableDisplayer(YearTokenDataMixin, ITrendDisplayer):

    name: str = field(default="Table")

    def setup(self, *_, **__):
        return

    def plot(self, corpus: VectorizedCorpus, compiled_data: dict, **_):  # pylint: disable=unused-argument

        with self.output:
            df = pd.DataFrame(data=compiled_data)
            df = df[['year'] + [x for x in df.columns if x != 'year']].set_index('year')
            g = display_grid(df)
            IPython.display.display(g)
