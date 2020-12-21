from dataclasses import dataclass

import penelope.notebook.utility as notebook_utility
from penelope.utility import getLogger

from .gofs_gui import GoFsGUI
from .trends_data import TrendsData
from .trends_gui import TrendsGUI

logger = getLogger("penelope")

# debug_view = ipywidgets.Output(layout={'border': '1px solid black'})
# display(debug_view)


@dataclass
class GofTrendsGUI:
    """GUI component for combined display of tokens distributions and goodness of fit to uniform distribution."""

    trends_gui: TrendsGUI
    gofs_gui: GoFsGUI

    def layout(self) -> notebook_utility.OutputsTabExt:
        _layout = (
            notebook_utility.OutputsTabExt(["Trends", "GoF"], layout={'width': '100%'})
            .display_content(0, what=self.trends_gui.layout(), clear=True)
            .display_content(1, what=self.gofs_gui.layout(), clear=True)
        )
        return _layout

    def display(self, trends_data: TrendsData):
        # self.trends_data = trends_data
        self.gofs_gui.display(trends_data=trends_data)
        self.trends_gui.display(trends_data=trends_data)
