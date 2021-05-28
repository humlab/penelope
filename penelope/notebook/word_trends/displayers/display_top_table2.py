import IPython.display

from ...ipyaggrid_utility import display_grid
from ._compile_mixins import TopTokens2MixIn
from .interface import ITrendDisplayer


class TopTokensSimpleDisplayer(TopTokens2MixIn, ITrendDisplayer):
    def __init__(self, name: str = "TopTokens"):
        super().__init__(name=name)

    def setup(self, *_, **__):
        return

    def plot(self, *, plot_data: dict, category_name: str, **_):  # pylint: disable=unused-argument
        with self.output:
            g = display_grid(plot_data)
            IPython.display.display(g)
