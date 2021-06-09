from typing import Union

import ipywidgets as widgets
import penelope.pipeline as pipeline
import penelope.workflows as workflows
from IPython.core.display import display
from penelope.corpus import VectorizedCorpus

from .. import dtm as dtm_gui
from .. import interface, utility
from .displayers import DEFAULT_WORD_TREND_DISPLAYERS
from .gof_and_trends_gui import GofTrendsGUI
from .gofs_gui import GoFsGUI
from .interface import TrendsData
from .trends_gui import TrendsGUI

view = widgets.Output(layout={'border': '2px solid green'})

LAST_ARGS = None
LAST_CORPUS_CONFIG = None
LAST_CORPUS = None


@view.capture(clear_output=utility.CLEAR_OUTPUT)
def loaded_callback(
    corpus: VectorizedCorpus,
    corpus_folder: str,
    corpus_tag: str,
):
    global LAST_CORPUS
    LAST_CORPUS = corpus
    trends_data: TrendsData = TrendsData(
        corpus=corpus,
        corpus_folder=corpus_folder,
        corpus_tag=corpus_tag,
        # FIXME #88 Review use of hard-coded value `n_count`
        n_count=25000,
    )

    gui: GofTrendsGUI = GofTrendsGUI(
        gofs_gui=GoFsGUI().setup(),
        trends_gui=TrendsGUI().setup(displayers=DEFAULT_WORD_TREND_DISPLAYERS),
    )

    display(gui.layout())
    gui.display(trends_data=trends_data)


@view.capture(clear_output=utility.CLEAR_OUTPUT)
def computed_callback(
    corpus: VectorizedCorpus,
    opts: interface.ComputeOpts,
) -> None:

    if opts.dry_run:
        return

    loaded_callback(corpus=corpus, corpus_folder=opts.target_folder, corpus_tag=opts.corpus_tag)


@view.capture(clear_output=utility.CLEAR_OUTPUT)
def compute_callback(args: interface.ComputeOpts, corpus_config: pipeline.CorpusConfig) -> VectorizedCorpus:
    global LAST_ARGS, LAST_CORPUS_CONFIG
    LAST_ARGS = args
    LAST_CORPUS_CONFIG = corpus_config
    if args.dry_run:
        print(args.command_line("vectorize_corpus"))
        return None

    corpus: VectorizedCorpus = workflows.document_term_matrix.compute(args=args, corpus_config=corpus_config)
    return corpus


def create_to_dtm_gui(
    *,
    corpus_folder: str,
    data_folder: str,
    corpus_config: Union[pipeline.CorpusConfig, str],
    resources_folder: str = None,
) -> widgets.CoreWidget:

    resources_folder = resources_folder or corpus_folder
    config: pipeline.CorpusConfig = (
        pipeline.CorpusConfig.find(corpus_config, resources_folder).folders(corpus_folder)
        if isinstance(corpus_config, str)
        else corpus_config
    )
    gui_compute: dtm_gui.ComputeGUI = dtm_gui.create_compute_gui(
        corpus_folder=corpus_folder,
        data_folder=data_folder,
        corpus_config=config,
        compute_callback=compute_callback,
        done_callback=computed_callback,
    )

    gui_load: dtm_gui.LoadGUI = dtm_gui.create_load_gui(
        corpus_folder=data_folder,
        loaded_callback=loaded_callback,
    )

    accordion = widgets.Accordion(
        children=[
            widgets.VBox(
                [
                    gui_load.layout(),
                ],
                layout={'border': '1px solid black', 'padding': '16px', 'margin': '4px'},
            ),
            widgets.VBox(
                [
                    gui_compute.layout(),
                ],
                layout={'border': '1px solid black', 'padding': '16px', 'margin': '4px'},
            ),
        ]
    )

    accordion.set_title(0, "LOAD AN EXISTING DOCUMENT-TERM MATRIX")
    accordion.set_title(1, '...OR COMPUTE A NEW DOCUMENT-TERM MATRIX')

    return widgets.VBox([accordion, view])
