from dataclasses import dataclass

import penelope.notebook.utility as notebook_utility
import penelope.notebook.word_trends as word_trends
from penelope.notebook.ipyaggrid_utility import display_grid
from penelope.notebook.word_trends.trends_data import TrendsData
from penelope.utility import getLogger

logger = getLogger()


@dataclass
class ExploreGUI:

    trends_data: TrendsData = None

    tab_main: notebook_utility.OutputsTabExt = None

    trends_gui: word_trends.TrendsGUI = None
    gofs_gui: word_trends.GoFsGUI = None

    def setup(self) -> "ExploreGUI":

        self.tab_main = notebook_utility.OutputsTabExt(["Data", "Trends", "Options", "GoF"], layout={'width': '98%'})
        self.trends_gui = word_trends.TrendsGUI().setup()
        self.gofs_gui = word_trends.GoFsGUI().setup()

        return self

    def display(self, trends_data: TrendsData) -> "ExploreGUI":

        self.trends_data = trends_data

        self.trends_gui.display(trends_data=trends_data)
        self.gofs_gui.display(trends_data=trends_data)

        self.tab_main.display_fx_result(0, display_grid, trends_data.memory.get('co_occurrences'), clear=True)
        self.tab_main.display_as_yaml(2, trends_data.compute_options, clear=True, width='800px', height='600px')

        return self

    def layout(self):
        layout = self.tab_main.display_content(1, what=self.trends_gui.layout(), clear=True).display_content(
            3, self.gofs_gui.layout(), clear=True
        )
        return layout
