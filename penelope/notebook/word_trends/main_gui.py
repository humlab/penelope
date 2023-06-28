from os.path import dirname, isfile
from typing import Union

import ipywidgets as widgets
from IPython.display import display

import penelope.pipeline as pipeline
from penelope.corpus import VectorizedCorpus
from penelope.utility import PivotKeys
from penelope.workflows import interface
from penelope.workflows.vectorize import dtm as workflow

from .. import dtm as dtm_gui
from .. import utility
from .displayers import DEFAULT_WORD_TREND_DISPLAYERS
from .interface import TrendsService
from .trends_gui import TrendsGUI

view = widgets.Output(layout={'border': '2px solid green'})

# pylint: disable=unused-argument


@view.capture(clear_output=utility.CLEAR_OUTPUT)
def loaded_callback(corpus: VectorizedCorpus, folder: str, n_top: int = 25000):
    trends_service: TrendsService = TrendsService(corpus=corpus, n_top=n_top)

    # gui: GofTrendsGUI = GofTrendsGUI(
    #     gofs_gui=GoFsGUI().setup(),
    #     trends_gui=TrendsGUI().setup(displayers=DEFAULT_WORD_TREND_DISPLAYERS),
    # )
    pivot_keys: PivotKeys = PivotKeys.load(folder) if isfile(folder) else None
    gui: TrendsGUI = TrendsGUI(pivot_key_specs=pivot_keys).setup(displayers=DEFAULT_WORD_TREND_DISPLAYERS)
    display(gui.layout())
    gui.display(trends_service=trends_service)


@view.capture(clear_output=utility.CLEAR_OUTPUT)
def computed_callback(corpus: VectorizedCorpus, opts: interface.ComputeOpts) -> None:
    if opts.dry_run:
        return

    loaded_callback(corpus=corpus, folder=dirname(opts.corpus_source))


@view.capture(clear_output=utility.CLEAR_OUTPUT)
def compute_callback(args: interface.ComputeOpts, corpus_config: pipeline.CorpusConfig) -> VectorizedCorpus:
    if args.dry_run:
        print(args.command_line("PYTHONPATH=. python ./penelope/scripts/dtm/vectorize.py"))
        return None
    corpus: VectorizedCorpus = workflow.compute(args=args, corpus_config=corpus_config)
    return corpus


def create_to_dtm_gui(
    *,
    corpus_folder: str,
    data_folder: str,
    resources_folder: str = None,
    corpus_config: Union[pipeline.CorpusConfig, str] = None,
) -> widgets.CoreWidget:
    resources_folder = resources_folder or corpus_folder
    config: pipeline.CorpusConfig = (
        None
        if corpus_config is None
        else pipeline.CorpusConfig.find(corpus_config, resources_folder).folders(corpus_folder)
        if isinstance(corpus_config, str)
        else corpus_config
    )
    gui_compute: dtm_gui.ComputeGUI = dtm_gui.create_compute_gui(
        corpus_folder=corpus_folder,
        data_folder=data_folder,
        config=config,
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
