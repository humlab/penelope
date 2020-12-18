# type: ignore
from ._displayer import ITrendDisplayer
from .display_bar import BarDisplayer
from .display_line import LineDisplayer
from .display_table import TableDisplayer

WORD_TREND_DISPLAYERS = [TableDisplayer, LineDisplayer, BarDisplayer]
