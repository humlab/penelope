import os

import ipywidgets
import penelope.common.goodness_of_fit as gof
import penelope.notebook.utility as notebook_utility
from penelope.corpus import VectorizedCorpus
from penelope.notebook.ipyaggrid_utility import display_grid
from penelope.utility import getLogger

from .displayers import WordTrendData
from .word_trends_gui import word_trend_gui

logger = getLogger("penelope")


def update_trend_data(
    *,
    corpus: VectorizedCorpus = None,
    corpus_folder: str = None,
    corpus_tag: str = None,
    n_count: int = 25000,
) -> WordTrendData:
    data = WordTrendData()
    data.corpus = corpus
    data.corpus_folder = corpus_folder
    data.corpus_tag = corpus_tag
    data.compute_options = VectorizedCorpus.load_options(tag=corpus_tag, folder=corpus_folder)
    data.goodness_of_fit = gof.compute_goddness_of_fits_to_uniform(
        data.corpus, None, verbose=True, metrics=['l2_norm', 'slope']
    )
    data.most_deviating_overview = gof.compile_most_deviating_words(data.goodness_of_fit, n_count=n_count)
    data.most_deviating = gof.get_most_deviating_words(
        data.goodness_of_fit, 'l2_norm', n_count=n_count, ascending=False, abs_value=True
    )
    return data


def build_layout(
    data: WordTrendData,  # pylint: disable=redefined-outer-name
):
    tab_trends = word_trend_gui(trend_data=data, display_widgets=False)

    tab_gof = (
        notebook_utility.OutputsTabExt(["GoF", "GoF (abs)", "Plots", "Slopes"])
        .display_fx_result(0, display_grid, data.goodness_of_fit)
        .display_fx_result(1, display_grid, data.most_deviating_overview[['l2_norm_token', 'l2_norm', 'abs_l2_norm']])
        .display_fx_result(2, gof.plot_metrics, data.goodness_of_fit, plot=False, lazy=True)
        .display_fx_result(
            3, gof.plot_slopes, data.corpus, data.most_deviating, "l2_norm", 600, 600, plot=False, lazy=True
        )
    )

    logger.info("Plotting...")
    layout = (
        notebook_utility.OutputsTabExt(["Trends", "GoF"])
        .display_content(0, what=tab_trends, clear=True)
        .display_content(1, what=tab_gof, clear=True)
    )

    return layout


def display_word_trends(
    *,
    corpus: VectorizedCorpus = None,
    corpus_folder: str = None,
    corpus_tag: str = None,
    output: ipywidgets.Output = None,
    **kwargs,
):

    with output:
        try:

            output.clear_output()

            if os.environ.get('VSCODE_LOGS', None) is not None:
                logger.error("bug-check: vscode detected, aborting plot...")
                return

            if corpus is None:
                logger.info("Please wait, loading corpus...")
                corpus = VectorizedCorpus.load(tag=corpus_tag, folder=corpus_folder)

            corpus = corpus.group_by_year()

            output.clear_output()

            logger.info("Please wait, compiling data...")

            data = update_trend_data(
                corpus=corpus,
                corpus_folder=corpus_folder,
                corpus_tag=corpus_tag,
                n_count=kwargs.get('n_count', 25000),
                **kwargs,
            )

            logger.info("Done!")
            build_layout(data=data).display()

        except gof.GoodnessOfFitComputeError as ex:
            logger.info(f"Unable to compute GoF: {str(ex)}")
        except Exception as ex:
            logger.exception(ex)
