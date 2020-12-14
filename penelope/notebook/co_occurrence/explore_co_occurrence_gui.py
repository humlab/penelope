from dataclasses import dataclass

import penelope.common.goodness_of_fit as gof
import penelope.notebook.utility as notebook_utility
import penelope.notebook.word_trends as word_trends
from penelope.notebook.ipyaggrid_utility import display_grid
from penelope.utility import getLogger

from .co_occurrence_data import CoOccurrenceData

logger = getLogger()


@dataclass
class GUI:

    data: CoOccurrenceData

    def layout(self):
        d: CoOccurrenceData = self.data
        tab_trends = word_trends.word_trends_pick_gui(d.corpus, tokens=d.most_deviating, display_widgets=False)

        tab_gof = (
            notebook_utility.OutputsTabExt(["GoF", "GoF (abs)", "Plots", "Slopes"])
            .display_fx_result(0, display_grid, d.goodness_of_fit)
            .display_fx_result(1, display_grid, d.most_deviating_overview[['l2_norm_token', 'l2_norm', 'abs_l2_norm']])
            .display_fx_result(2, gof.plot_metrics, d.goodness_of_fit, plot=False, lazy=True)
            .display_fx_result(
                3, gof.plot_slopes, d.corpus, d.most_deviating, "l2_norm", 600, 600, plot=False, lazy=True
            )
        )

        layout = (
            notebook_utility.OutputsTabExt(["Data", "Explore", "Options", "GoF"])
            .display_fx_result(0, display_grid, d.co_occurrences)
            .display_content(1, what=tab_trends, clear=True)
            .display_as_yaml(2, d.compute_options, clear=True, width='800px', height='600px')
            .display_content(3, tab_gof, clear=True)
        )

        return layout


def create_gui(data: CoOccurrenceData) -> "GUI":

    # if os.environ.get('VSCODE_LOGS', None) is not None:
    #     logger.error("bug-check: vscode detected, aborting plot...")
    #     return

    gui = GUI(data)

    return gui
