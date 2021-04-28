# type: ignore
# isort: skip_file
# pylint: disable=wrong-import-position)
import logging

logging.getLogger("matplotlib").setLevel(level=logging.ERROR)

from ._compile_mixins import CategoryDataMixin, LinesDataMixin, PenelopeBugCheck
from ._displayer import ITrendDisplayer
from .display_bar import BarDisplayer
from .display_line import LineDisplayer
from .display_network import NetworkDisplayer, create_network
from .display_table import TableDisplayer, UnnestedExplodeTableDisplayer, UnnestedTableDisplayer
from .display_top_table import TopTokensDisplayer


DEFAULT_WORD_TREND_DISPLAYERS = [
    TableDisplayer,
    UnnestedExplodeTableDisplayer,
    LineDisplayer,
    BarDisplayer,
    TopTokensDisplayer,
    # NetworkDisplayer,
]
