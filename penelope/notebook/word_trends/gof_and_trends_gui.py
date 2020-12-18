from dataclasses import dataclass
from typing import List

import penelope.notebook.utility as notebook_utility
from penelope.corpus import VectorizedCorpus
from penelope.utility import getLogger

from .gofs_gui import GoFsGUI
from .word_trend_data import WordTrendData
from .word_trends_gui import TrendsGUI

logger = getLogger("penelope")

# debug_view = ipywidgets.Output(layout={'border': '1px solid black'})
# display(debug_view)


@dataclass
class GofTrendsGUI:
    """GUI component for combined display of tokens distributions and goodness of fit to uniform distribution."""

    trend_data: WordTrendData

    trends_gui: TrendsGUI
    gofs_gui: GoFsGUI

    def layout(self) -> notebook_utility.OutputsTabExt:
        _layout = (
            notebook_utility.OutputsTabExt(["Trends", "GoF"])
            .display_content(0, what=self.trends_gui.layout(), clear=True)
            .display_content(1, what=self.gofs_gui.layout(), clear=True)
        )
        return _layout

    def display(self, trend_data: WordTrendData, corpus: VectorizedCorpus, indices: List[int]):
        self.trend_data = trend_data
        self.gofs_gui.display(trend_data=trend_data)
        self.trends_gui.display(trend_data=trend_data, corpus=corpus, indices=indices)


def update_plot(gui: TrendsGUI, trend_data: WordTrendData):

    indices = trend_data.find_indices(gui.words, gui.word_count, group_by=gui.group_by, normalize=gui.normalize)
    gui.display(trend_data.corpus, indices=indices)


def create_gof_and_trends_gui(trend_data: WordTrendData):

    gui = GofTrendsGUI(
        trend_data=trend_data,
        gofs_gui=GoFsGUI().setup(),
        trends_gui=TrendsGUI().setup(),
    )

    return gui
