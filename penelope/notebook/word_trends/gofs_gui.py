import penelope.common.goodness_of_fit as gof

from .. import grid_utility as gu
from .. import utility as notebook_utility
from .interface import TrendsData


class GoFsGUI:
    """GUI component for displaying token distributions goodness of fit to uniform distribution."""

    def __init__(self, tab_gof: notebook_utility.OutputsTabExt = None):
        self.tab_gof: notebook_utility.OutputsTabExt = tab_gof
        self.is_displayed: bool = False

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
        gof_data: gof.GofData = trends_data.gof_data
        self.tab_gof = (
            self.tab_gof.display_fx_result(0, gu.table_widget, gof_data.goodness_of_fit)
            .display_fx_result(
                1, gu.table_widget, gof_data.most_deviating_overview[['l2_norm_token', 'l2_norm', 'abs_l2_norm']]
            )
            .display_fx_result(2, gof.plot_metrics, gof_data.goodness_of_fit, plot=False, lazy=True)
            .display_fx_result(
                3,
                gof.plot_slopes,
                trends_data.corpus,
                gof_data.most_deviating,
                "l2_norm",
                600,
                600,
                plot=False,
                lazy=True,
            )
        )
        self.is_displayed = True
        return self
