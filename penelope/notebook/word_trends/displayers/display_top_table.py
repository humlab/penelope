from dataclasses import dataclass, field
from typing import Any

import IPython.display
from penelope.corpus.dtm.vectorized_corpus import VectorizedCorpus
from penelope.notebook.ipyaggrid_utility import display_grid

from ._displayer import ITrendDisplayer


@dataclass
class TopTokensDisplayer(ITrendDisplayer):

    name: str = field(default="TopTokens")

    def setup(self, *_, **__):
        return

    def compile(self, *, corpus: VectorizedCorpus, **__) -> Any:
        top_terms = corpus.get_top_terms(category_column='category', n_count=10000, kind='token+count')
        return top_terms

    def plot(self, plot_data: dict, **_):  # pylint: disable=unused-argument

        with self.output:
            g = display_grid(plot_data)
            IPython.display.display(g)
