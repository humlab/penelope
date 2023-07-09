from os.path import dirname, isdir, isfile
from typing import Any, Union

import ipywidgets as w

import penelope.corpus as pc
import penelope.pipeline as pp
import penelope.utility as pu
from penelope.workflows.vectorize import dtm as workflow

from .. import dtm as dtm_gui
from .. import pick_file_gui as pfg
from . import DEFAULT_WORD_TREND_DISPLAYERS
from .interface import TrendsService
from .trends_gui import TrendsGUI

# pylint: disable=unused-argument


def create_advanced_dtm_gui(
    *,
    corpus_folder: str,
    data_folder: str,
    resources_folder: str = None,
    corpus_config: Union[pp.CorpusConfig, str] = None,
) -> w.CoreWidget:
    content_placeholder: w.VBox = w.VBox()

    def display_trends(*, result: str | pc.VectorizedCorpus, folder: str = None, sender: Any = None, **args) -> None:
        nonlocal content_placeholder
        corpus: pc.VectorizedCorpus = pc.VectorizedCorpus.load(filename=result) if isinstance(result, str) else result

        trends_service: TrendsService = TrendsService(corpus=corpus, n_top=25000)
        # gui: GofTrendsGUI = GofTrendsGUI(
        #     gofs_gui=GoFsGUI().setup(),
        #     trends_gui=TrendsGUI().setup(displayers=DEFAULT_WORD_TREND_DISPLAYERS),
        # )
        pivot_keys: pu.PivotKeys = pu.PivotKeys.load(folder) if isfile(folder) else None
        gui: TrendsGUI = TrendsGUI(pivot_key_specs=pivot_keys).setup(displayers=DEFAULT_WORD_TREND_DISPLAYERS)
        content_placeholder.children = [gui.layout()]
        gui.display(trends_service=trends_service)

    resources_folder = resources_folder or corpus_folder
    config: pp.CorpusConfig = (
        None
        if corpus_config is None
        else pp.CorpusConfig.find(corpus_config, resources_folder).folders(corpus_folder)
        if isinstance(corpus_config, str)
        else corpus_config
    )
    gui_compute: dtm_gui.ComputeGUI = dtm_gui.create_compute_gui(
        corpus_folder=corpus_folder,
        data_folder=data_folder,
        config=config,
        compute_callback=workflow.compute,
        done_callback=display_trends,
    )

    gui_load: pfg.PickFileGUI = pfg.PickFileGUI(
        folder=data_folder, pattern='*_vector_data.npz', picked_callback=display_trends
    )

    accordion = w.Accordion(
        children=[
            w.VBox(
                [gui_load.layout()],
                layout={'border': '1px solid black', 'padding': '16px', 'margin': '4px'},
            ),
            w.VBox(
                [gui_compute.layout()],
                layout={'border': '1px solid black', 'padding': '16px', 'margin': '4px'},
            ),
        ]
    )

    accordion.set_title(0, "LOAD AN EXISTING DOCUMENT-TERM MATRIX")
    accordion.set_title(1, '...OR COMPUTE A NEW DOCUMENT-TERM MATRIX')

    return w.VBox([accordion, content_placeholder])


def create_simple_dtm_gui(folder: str) -> pfg.PickFileGUI:
    def display_callback(filename: str, sender: pfg.PickFileGUI):
        if not pc.VectorizedCorpus.is_dump(filename):
            raise ValueError(f"Expected a DTM file, got {filename or 'None'}")

        corpus: pc.VectorizedCorpus = pc.VectorizedCorpus.load(filename=filename)
        folder: str = dirname(filename)
        trends_service: TrendsService = TrendsService(corpus=corpus, n_top=25000)
        pivot_keys: pu.PivotKeys = pu.PivotKeys.load(folder) if isdir(folder) else None
        gui: TrendsGUI = TrendsGUI(pivot_key_specs=pivot_keys).setup(displayers=DEFAULT_WORD_TREND_DISPLAYERS)
        sender.add(gui.layout())
        gui.display(trends_service=trends_service)

    gui: pfg.PickFileGUI = pfg.PickFileGUI(
        folder=folder, pattern='*_vector_data.npz', picked_callback=display_callback, kind='picker'
    ).setup()
    return gui
