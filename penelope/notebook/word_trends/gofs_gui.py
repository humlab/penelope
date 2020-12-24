from dataclasses import dataclass

import penelope.common.goodness_of_fit as gof
import penelope.notebook.utility as notebook_utility
from penelope.notebook.ipyaggrid_utility import display_grid
from penelope.utility import getLogger

from .trends_data import TrendsData

logger = getLogger("penelope")


@dataclass
class GoFsGUI:
    """GUI component for displaying token distributions goodness of fit to uniform distribution."""

    tab_gof: notebook_utility.OutputsTabExt = None
    is_displayed: bool = False

    def setup(self) -> "GoFsGUI":
        self.tab_gof: notebook_utility.OutputsTabExt = notebook_utility.OutputsTabExt(
            ["GoF", "GoF (abs)", "Plots", "Slopes"], layout={'width': '98%'}
        )
        return self

    def layout(self) -> notebook_utility.OutputsTabExt:
        return self.tab_gof

    def display(self, trends_data: TrendsData) -> "GoFsGUI":
        if self.is_displayed:
            return self
        self.tab_gof = (
            self.tab_gof.display_fx_result(0, display_grid, trends_data.goodness_of_fit)
            .display_fx_result(
                1, display_grid, trends_data.most_deviating_overview[['l2_norm_token', 'l2_norm', 'abs_l2_norm']]
            )
            .display_fx_result(2, gof.plot_metrics, trends_data.goodness_of_fit, plot=False, lazy=True)
            .display_fx_result(
                3,
                gof.plot_slopes,
                trends_data.corpus,
                trends_data.most_deviating,
                "l2_norm",
                600,
                600,
                plot=False,
                lazy=True,
            )
        )
        self.is_displayed = True
        return self
