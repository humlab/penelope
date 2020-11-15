from ._displayer import ITrendDisplayer, WordTrendData
from .display_bar import BarDisplayer
from .display_line import LineDisplayer
from .display_table import TableDisplayer

DISPLAYERS = [TableDisplayer, LineDisplayer, BarDisplayer]
