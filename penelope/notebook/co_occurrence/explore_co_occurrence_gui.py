from dataclasses import dataclass

import penelope.common.goodness_of_fit as gof
import penelope.notebook.utility as notebook_utility
import penelope.notebook.word_trends as word_trends
from penelope.notebook.ipyaggrid_utility import display_grid
from penelope.notebook.word_trends.trends_data import TrendsData
from penelope.utility import getLogger

logger = getLogger()


@dataclass
class ExploreCoOccurrencesGUI:

    trends_data: TrendsData

    def layout(self):

        tab_gof = (
            notebook_utility.OutputsTabExt(["GoF", "GoF (abs)", "Plots", "Slopes"], layout={'width': '100%'})
            .display_fx_result(0, display_grid, self.trends_data.goodness_of_fit)
            .display_fx_result(
                1, display_grid, self.trends_data.most_deviating_overview[['l2_norm_token', 'l2_norm', 'abs_l2_norm']]
            )
            .display_fx_result(2, gof.plot_metrics, self.trends_data.goodness_of_fit, plot=False, lazy=True)
            .display_fx_result(
                3,
                gof.plot_slopes,
                self.trends_data.corpus,
                self.trends_data.most_deviating,
                "l2_norm",
                600,
                600,
                plot=False,
                lazy=True,
            )
        )

        gof_and_trends_gui = word_trends.GofTrendsGUI(
            gofs_gui=word_trends.GoFsGUI().setup(),
            trends_gui=word_trends.TrendsGUI().setup(),
        )

        trends_with_pick_gui = word_trends.TrendsWithPickTokensGUI(
            self.trends_data.corpus, tokens=self.trends_data.most_deviating, display_widgets=False
        )

        layout = (
            notebook_utility.OutputsTabExt(["Data", "Trends", "Explore", "Options", "GoF"], layout={'width': '100%'})
            .display_fx_result(0, display_grid, self.trends_data.memory.get('co_occurrences'))
            .display_content(1, what=gof_and_trends_gui.layout(), clear=True)
            .display_content(2, what=trends_with_pick_gui.layout(), clear=True)
            .display_as_yaml(3, self.trends_data.compute_options, clear=True, width='800px', height='600px')
            .display_content(4, tab_gof, clear=True)
        )

        return layout
