# type: ignore
# isort: skip_file
# pylint: disable=wrong-import-position

import logging

logging.getLogger("matplotlib").setLevel(level=logging.ERROR)

from ._compile_mixins import CategoryDataMixin, LinesDataMixin, PenelopeBugCheck  # noqa: E402
from ._displayer import ITrendDisplayer  # noqa: E402
from .display_bar import BarDisplayer  # noqa: E402
from .display_line import LineDisplayer  # noqa: E402
from .display_network import NetworkDisplayer, create_network  # noqa: E402
from .display_table import TableDisplayer, UnnestedExplodeTableDisplayer, UnnestedTableDisplayer  # noqa: E402
from .display_top_table import TopTokensDisplayer  # noqa: E402


DEFAULT_WORD_TREND_DISPLAYERS = [
    TableDisplayer,
    UnnestedExplodeTableDisplayer,
    LineDisplayer,
    BarDisplayer,
    # NetworkDisplayer,
]
