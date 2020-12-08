from dataclasses import dataclass

import penelope.common.goodness_of_fit as gof
import penelope.notebook.utility as notebook_utility
from penelope.corpus import VectorizedCorpus
from penelope.notebook.ipyaggrid_utility import display_grid
from penelope.utility import getLogger

from .displayers import WordTrendData
from .word_trends_tabs_gui import create_tabs_gui

logger = getLogger("penelope")

# debug_view = ipywidgets.Output(layout={'border': '1px solid black'})
# display(debug_view)


@dataclass
class WordTrendsGUI:

    word_trend_data: WordTrendData

    def layout(self):
        data = self.word_trend_data
        tab_gui = create_tabs_gui(trend_data=data)
        tab_gof = (
            notebook_utility.OutputsTabExt(["GoF", "GoF (abs)", "Plots", "Slopes"])
            .display_fx_result(0, display_grid, data.goodness_of_fit)
            .display_fx_result(
                1, display_grid, data.most_deviating_overview[['l2_norm_token', 'l2_norm', 'abs_l2_norm']]
            )
            .display_fx_result(2, gof.plot_metrics, data.goodness_of_fit, plot=False, lazy=True)
            .display_fx_result(
                3, gof.plot_slopes, data.corpus, data.most_deviating, "l2_norm", 600, 600, plot=False, lazy=True
            )
        )
        _layout = (
            notebook_utility.OutputsTabExt(["Trends", "GoF"])
            .display_content(0, what=tab_gui.layout(), clear=True)
            .display_content(1, what=tab_gof, clear=True)
        )

        return _layout


def create_gui(
    *,
    corpus: VectorizedCorpus = None,
    corpus_folder: str = None,
    corpus_tag: str = None,
    word_trend_data: WordTrendData = None,
    **kwargs,
):
    if corpus is None:
        logger.info("Please wait, loading corpus...")
        corpus = VectorizedCorpus.load(tag=corpus_tag, folder=corpus_folder)
    corpus = corpus.group_by_year()

    try:

        word_trend_data = word_trend_data or WordTrendData().update(
            corpus=corpus,
            corpus_folder=corpus_folder,
            corpus_tag=corpus_tag,
            n_count=kwargs.get('n_count', 25000),
            **kwargs,
        )

        gui = WordTrendsGUI(word_trend_data=word_trend_data)

        return gui

    except gof.GoodnessOfFitComputeError as ex:
        logger.info(f"Unable to compute GoF: {str(ex)}")
        raise
    except Exception as ex:
        logger.exception(ex)
        raise
