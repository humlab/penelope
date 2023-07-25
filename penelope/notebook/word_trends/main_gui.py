from os.path import dirname, isdir, isfile
from typing import Any, Self, Union

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


class ComplexTrendsGUI:
    def __init__(
        self,
        *,
        corpus_folder: str,
        data_folder: str,
        resources_folder: str = None,
        corpus_config: Union[pp.CorpusConfig, str] = None,
        n_top: int = 25000,
    ) -> w.CoreWidget:
        self.content_placeholder: w.VBox = w.VBox()
        resources_folder = resources_folder or corpus_folder
        self.config: pp.CorpusConfig = (
            None
            if corpus_config is None
            else pp.CorpusConfig.find(corpus_config, resources_folder).folders(corpus_folder)
            if isinstance(corpus_config, str)
            else corpus_config
        )
        self.gui_compute: dtm_gui.ComputeGUI = dtm_gui.ComputeGUI(
            default_corpus_path=corpus_folder,
            default_data_folder=data_folder,
        )
        self.gui_load: pfg.PickFileGUI = pfg.PickFileGUI(
            folder=data_folder, pattern='*_vector_data.npz', picked_callback=self.picked_callback
        )
        self.gui_trends: TrendsGUI = TrendsGUI(pivot_key_specs={})
        self.trends_service: TrendsService = TrendsService(corpus=None, n_top=n_top)

    def setup(self) -> Self:
        self.gui_compute.setup(
            config=self.config,
            compute_callback=workflow.compute,
            done_callback=self.display_trends,
        )
        self.gui_load.setup()
        self.gui_trends.setup(displayers=DEFAULT_WORD_TREND_DISPLAYERS)
        return self

    def layout(self) -> w.CoreWidget:
        accordion = w.Accordion(
            children=[
                w.VBox(
                    [self.gui_load.layout()],
                    layout={'border': '1px solid black', 'padding': '16px', 'margin': '4px'},
                ),
                w.VBox(
                    [self.gui_compute.layout()],
                    layout={'border': '1px solid black', 'padding': '16px', 'margin': '4px'},
                ),
            ]
        )

        accordion.set_title(0, "LOAD AN EXISTING DOCUMENT-TERM MATRIX")
        accordion.set_title(1, '...OR COMPUTE A NEW DOCUMENT-TERM MATRIX')

        return w.VBox([accordion, self.content_placeholder])

    def picked_callback(self, filename: str, sender: Any = None, **args) -> None:
        self.display_trends(result=filename, folder=dirname(filename), sender=sender, **args)

    def display_trends(
        self, *, result: str | pc.VectorizedCorpus, folder: str = None, sender: Any = None, **args
    ) -> None:
        self.trends_service.corpus = pc.VectorizedCorpus.load(filename=result) if isinstance(result, str) else result
        self.gui_trends.pivot_keys = pu.PivotKeys.load(folder) if isfile(folder) else None
        self.content_placeholder.children = [self.gui_trends.layout()]
        self.gui_trends.display(trends_service=self.trends_service)


class SimpleTrendsGUI:
    def __init__(self, folder: str):
        self.folder: str = folder
        self.gui_pick: pfg.PickFileGUI = None
        self.gui_trends: TrendsGUI = None

    def display_callback(self, filename: str, sender: pfg.PickFileGUI):
        if not pc.VectorizedCorpus.is_dump(filename):
            raise ValueError(f"Expected a DTM file, got {filename or 'None'}")

        folder: str = dirname(filename)

        sender.payload.corpus = pc.VectorizedCorpus.load(filename=filename)
        sender.payload.pivot_key_specs = pu.PivotKeys.load(folder) if isdir(folder) else None
        sender.payload.display(trends_service=sender.payload)

    def setup(self) -> Self:
        self.gui_pick: pfg.PickFileGUI = pfg.PickFileGUI(
            folder=self.folder, pattern='*_vector_data.npz', picked_callback=self.display_callback, kind='picker'
        ).setup()

        self.gui_trends: TrendsGUI = TrendsGUI(pivot_key_specs=None).setup(displayers=DEFAULT_WORD_TREND_DISPLAYERS)
        self.gui_trends.trends_service = TrendsService(corpus=None, n_top=25000)

        self.gui_pick.payload = self.gui_trends
        self.gui_pick.add(self.gui_trends.layout())

        return self

    def layout(self) -> w.CoreWidget:
        return self.gui_pick.layout()
