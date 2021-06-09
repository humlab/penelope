from loguru import logger
from penelope import co_occurrence
from penelope.notebook.word_trends.displayers.display_top_table import CoOccurrenceTopTokensDisplayer

from .. import utility as notebook_utility
from .. import word_trends
from ..co_occurrence.tabular_gui import TabularCoOccurrenceGUI


class ExploreGUI:
    def __init__(self, bundle: co_occurrence.Bundle):
        self.bundle: co_occurrence.Bundle = bundle
        self.trends_data: word_trends.BundleTrendsData = None
        self.tab_main: notebook_utility.OutputsTabExt = None
        self.trends_gui: word_trends.TrendsGUI = None
        self.gofs_gui: word_trends.GoFsGUI = None
        self.gofs_enabled: bool = False

    def setup(self) -> "ExploreGUI":
        self.tab_main = notebook_utility.OutputsTabExt(
            ["Data", "Trends", "Options", "GoF", "TopTokens"], layout={'width': '98%'}
        )
        self.trends_gui = word_trends.TrendsGUI().setup(displayers=word_trends.DEFAULT_WORD_TREND_DISPLAYERS)
        self.gofs_gui = word_trends.GoFsGUI().setup() if self.gofs_enabled else None

        return self

    def display(self, trends_data: word_trends.BundleTrendsData) -> "ExploreGUI":

        try:
            self.trends_data = trends_data

            self.trends_gui.display(trends_data=trends_data)

            if self.gofs_gui:
                self.gofs_gui.display(trends_data=trends_data)

            # self.tab_main.display_fx_result(0, display_grid, trends_data.memory.get('co_occurrences'), clear=True)
            # self.tab_main.display_fx_result(
            #     0, display_table, self.trim_data(trends_data.memory.get('co_occurrences')), clear=True
            # )
            self.tab_main.display_content(0, TabularCoOccurrenceGUI(bundle=self.bundle).setup(), clear=True)
            self.tab_main.display_as_yaml(2, self.bundle.compute_options, clear=True, width='800px', height='600px')

            top_displayer: CoOccurrenceTopTokensDisplayer = CoOccurrenceTopTokensDisplayer(bundle=self.bundle).setup()
            self.tab_main.display_content(4, top_displayer.layout(), clear=True)
        except KeyError as ex:
            logger.error(
                f"KeyError: {str(ex)}, columns in data: {' '.join(trends_data.transformed_corpus.document_index.columns)}"
            )
            raise

        return self

    def layout(self) -> notebook_utility.OutputsTabExt:
        layout: notebook_utility.OutputsTabExt = self.tab_main.display_content(
            1, what=self.trends_gui.layout(), clear=True
        )
        if self.gofs_gui:
            layout = layout.display_content(3, self.gofs_gui.layout(), clear=True)
        return layout
