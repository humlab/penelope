from penelope.utility import deprecated

from .. import utility as notebook_utility
from .displayers import DEFAULT_WORD_TREND_DISPLAYERS
from .gofs_gui import GoFsGUI
from .interface import TrendsData
from .trends_gui import TrendsGUI


@deprecated
class GofTrendsGUI:
    """GUI component for combined display of tokens distributions and goodness of fit to uniform distribution."""

    def __init__(self, trends_gui: TrendsGUI, gofs_gui: GoFsGUI):
        self.trends_gui: TrendsGUI = trends_gui
        self.gofs_gui: GoFsGUI = gofs_gui

    def setup(self) -> "GofTrendsGUI":
        self.trends_gui = TrendsGUI().setup(displayers=DEFAULT_WORD_TREND_DISPLAYERS)
        self.gofs_gui = GoFsGUI().setup()
        return self

    def layout(self) -> notebook_utility.OutputsTabExt:
        _layout = (
            notebook_utility.OutputsTabExt(["Trends", "GoF"], layout={'width': '98%'})
            .display_content(0, what=self.trends_gui.layout(), clear=True)
            .display_content(1, what=self.gofs_gui.layout(), clear=True)
        )
        return _layout

    def display(self, trends_data: TrendsData):
        self.gofs_gui.display(trends_data=trends_data)
        self.trends_gui.display(trends_data=trends_data)
