from dataclasses import dataclass

import pandas as pd
import penelope.notebook.utility as notebook_utility
import penelope.notebook.word_trends as word_trends
from penelope.notebook.co_occurrence.display_data import CoOccurrenceTable
from penelope.notebook.word_trends.trends_data import TrendsData
from penelope.utility import getLogger

logger = getLogger()


@dataclass
class ExploreGUI:

    trends_data: TrendsData = None

    tab_main: notebook_utility.OutputsTabExt = None

    trends_gui: word_trends.TrendsGUI = None
    gofs_gui: word_trends.GoFsGUI = None
    gofs_enabled: bool = False
    global_tokens_count_threshold: int = 25

    def setup(self) -> "ExploreGUI":

        self.tab_main = notebook_utility.OutputsTabExt(["Data", "Trends", "Options", "GoF"], layout={'width': '98%'})
        self.trends_gui = word_trends.TrendsGUI().setup()
        self.gofs_gui = word_trends.GoFsGUI().setup() if self.gofs_enabled else None

        return self

    def display(self, trends_data: TrendsData) -> "ExploreGUI":

        self.trends_data = trends_data

        self.trends_gui.display(trends_data=trends_data)

        if self.gofs_gui:
            self.gofs_gui.display(trends_data=trends_data)

        # self.tab_main.display_fx_result(0, display_grid, trends_data.memory.get('co_occurrences'), clear=True)
        # self.tab_main.display_fx_result(
        #     0, display_table, self.trim_data(trends_data.memory.get('co_occurrences')), clear=True
        # )

        data: pd.DataFrame = trends_data.memory.get('co_occurrences')

        self.tab_main.display_content(
            0, CoOccurrenceTable(data, global_tokens_count_threshold=self.global_tokens_count_threshold), clear=True
        )
        self.tab_main.display_as_yaml(2, trends_data.compute_options, clear=True, width='800px', height='600px')

        return self

    def layout(self) -> notebook_utility.OutputsTabExt:
        layout: notebook_utility.OutputsTabExt = self.tab_main.display_content(
            1, what=self.trends_gui.layout(), clear=True
        )
        if self.gofs_gui:
            layout = layout.display_content(3, self.gofs_gui.layout(), clear=True)
        return layout
