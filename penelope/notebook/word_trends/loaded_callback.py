import os
from dataclasses import dataclass
from typing import Dict

import ipywidgets
import pandas as pd
import penelope.common.goodness_of_fit as gof
import penelope.notebook.utility as notebook_utility
import penelope.notebook.word_trends.word_trends_gui as xxx_word_trends_gui
from penelope.common.goodness_of_fit import GoodnessOfFitComputeError
from penelope.corpus import VectorizedCorpus
from penelope.notebook.ipyaggrid_utility import display_grid
from penelope.utility import getLogger

logger = getLogger("penelope")


@dataclass
class State:
    corpus: VectorizedCorpus = None
    corpus_folder: str = None
    corpus_tag: str = None
    compute_options: Dict = None
    goodness_of_fit: pd.DataFrame = None
    most_deviating_overview: pd.DataFrame = None
    most_deviating: pd.DataFrame = None


def update_state(
    state: State,  # pylint: disable=redefined-outer-name
    *,
    corpus: VectorizedCorpus = None,
    corpus_folder: str = None,
    corpus_tag: str = None,
    n_count: int = 10000,
    **_,
):

    state.corpus = corpus
    state.corpus_folder = corpus_folder
    state.corpus_tag = corpus_tag

    state.goodness_of_fit = gof.compute_goddness_of_fits_to_uniform(state.corpus, None, verbose=False)
    state.most_deviating_overview = gof.compile_most_deviating_words(state.goodness_of_fit, n_count=n_count)
    state.most_deviating = gof.get_most_deviating_words(
        state.goodness_of_fit, 'l2_norm', n_count=n_count, ascending=False, abs_value=True
    )

    return state


def build_layout(
    state: State,  # pylint: disable=redefined-outer-name
):
    tab_trends = xxx_word_trends_gui.display_gui(state=state, display_widgets=False)

    tab_gof = (
        notebook_utility.OutputsTabExt(["GoF", "GoF (abs)", "Plots", "Slopes"])
        .display_fx_result(0, display_grid, state.goodness_of_fit)
        .display_fx_result(1, display_grid, state.most_deviating_overview[['l2_norm_token', 'l2_norm', 'abs_l2_norm']])
        .display_fx_result(2, gof.plot_metrics, state.goodness_of_fit, plot=False, lazy=True)
        .display_fx_result(
            3, gof.plot_slopes, state.corpus, state.most_deviating, "l2_norm", 600, 600, plot=False, lazy=True
        )
    )

    logger.info("Plotting...")
    layout = (
        notebook_utility.OutputsTabExt(["Trends", "GoF"])
        .display_content(0, what=tab_trends, clear=True)
        .display_content(1, what=tab_gof, clear=True)
    )

    return layout


state = State()


def loaded_callback(
    output: ipywidgets.Output,
    *,
    corpus: VectorizedCorpus = None,
    corpus_folder: str = None,
    corpus_tag: str = None,
    n_count: int = 10000,
    **kwargs,
):
    global state

    with output:
        try:

            output.clear_output()

            if os.environ.get('VSCODE_LOGS', None) is not None:
                logger.error("bug-check: vscode detected, aborting plot...")
                return

            if corpus is None:
                logger.info("Please wait, loading corpus...")
                # FIXME IS GROUP BY YEAR NEEDED?
                corpus = VectorizedCorpus.load(tag=corpus_tag, folder=corpus_folder).group_by_year()

            output.clear_output()

            _ = update_state(
                state,
                corpus=corpus,
                corpus_folder=corpus_folder,
                corpus_tag=corpus_tag,
                n_count=n_count,
                **kwargs,
            )

            build_layout(state=state).display()

        except GoodnessOfFitComputeError as ex:
            logger.info(f"Unable to compute GoF: {str(ex)}")
        except Exception as ex:
            logger.exception(ex)
