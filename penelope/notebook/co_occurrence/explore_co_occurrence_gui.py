from dataclasses import dataclass

import penelope.common.goodness_of_fit as gof
import penelope.notebook.utility as notebook_utility
import penelope.notebook.word_trends as word_trends
from penelope.notebook.ipyaggrid_utility import display_grid
from penelope.notebook.word_trends import gof_and_trends_gui
from penelope.notebook.word_trends.word_trend_data import WordTrendData
from penelope.utility import getLogger

logger = getLogger()


@dataclass
class ExploreCoOccurrencesGUI:

    trend_data: WordTrendData

    def layout(self):

        tab_gof = (
            notebook_utility.OutputsTabExt(["GoF", "GoF (abs)", "Plots", "Slopes"])
            .display_fx_result(0, display_grid, self.trend_data.goodness_of_fit)
            .display_fx_result(
                1, display_grid, self.trend_data.most_deviating_overview[['l2_norm_token', 'l2_norm', 'abs_l2_norm']]
            )
            .display_fx_result(2, gof.plot_metrics, self.trend_data.goodness_of_fit, plot=False, lazy=True)
            .display_fx_result(
                3,
                gof.plot_slopes,
                self.trend_data.corpus,
                self.trend_data.most_deviating,
                "l2_norm",
                600,
                600,
                plot=False,
                lazy=True,
            )
        )

        tab_trends = gof_and_trends_gui.create_gof_and_trends_gui(
            trend_data=self.trend_data,
        ).layout()

        tab_explore = word_trends.create_word_trends_pick_gui(
            self.trend_data.corpus, tokens=self.trend_data.most_deviating, display_widgets=False
        )

        layout = (
            notebook_utility.OutputsTabExt(["Data", "Trends", "Explore", "Options", "GoF"])
            .display_fx_result(0, display_grid, self.trend_data.memory.get('co_occurrences'))
            .display_content(1, what=tab_trends, clear=True)
            .display_content(2, what=tab_explore, clear=True)
            .display_as_yaml(3, self.trend_data.compute_options, clear=True, width='800px', height='600px')
            .display_content(4, tab_gof, clear=True)
        )

        return layout
