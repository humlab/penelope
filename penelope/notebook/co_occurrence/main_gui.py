from typing import Callable, Optional, Union

import ipywidgets as widgets
from IPython.display import display

from penelope import co_occurrence, pipeline
from penelope.workflows import ComputeOpts
from penelope.workflows import co_occurrence as workflow

from ..utility import CLEAR_OUTPUT
from ..word_trends.interface import BundleTrendsData
from . import explore_co_occurrence_gui as explore_gui
from . import load_co_occurrences_gui as load_gui
from . import to_co_occurrence_gui as compute_gui

view = widgets.Output(layout={'border': '2px solid green'})

LAST_ARGS = None
LAST_CONFIG = None


def create(
    data_folder: str,
    filename_pattern: str = co_occurrence.FILENAME_PATTERN,
    loaded_callback: Callable[[co_occurrence.Bundle], None] = None,
) -> load_gui.LoadGUI:

    gui: load_gui.LoadGUI = load_gui.LoadGUI(default_path=data_folder).setup(
        filename_pattern=filename_pattern,
        load_callback=co_occurrence.Bundle.load,
        loaded_callback=loaded_callback,
    )
    return gui


@view.capture(clear_output=CLEAR_OUTPUT)
def compute_co_occurrence_callback(
    corpus_config: pipeline.CorpusConfig,
    args: ComputeOpts,
    corpus_source: Optional[str] = None,
) -> co_occurrence.Bundle:
    try:
        global LAST_ARGS, LAST_CONFIG
        LAST_ARGS = args
        LAST_CONFIG = corpus_config

        if args.dry_run:
            print(args.command_line("co-occurrence"))
            return None

        bundle: co_occurrence.Bundle = workflow.compute(
            args=args,
            corpus_config=corpus_config,
            tagged_corpus_source=corpus_source,
        )
        return bundle
    except co_occurrence.ZeroComputeError:
        return None


class MainGUI:
    def __init__(
        self,
        corpus_config: Union[pipeline.CorpusConfig, str],
        corpus_folder: str,
        data_folder: str,
        resources_folder: str,
    ) -> widgets.VBox:

        self.bundle: co_occurrence.Bundle = None
        self.trends_data: BundleTrendsData = None
        self.config = (
            corpus_config
            if isinstance(corpus_config, pipeline.CorpusConfig)
            else pipeline.CorpusConfig.find(corpus_config, resources_folder).folders(corpus_folder)
        )

        self.gui_compute: compute_gui.ComputeGUI = compute_gui.create_compute_gui(
            corpus_folder=corpus_folder,
            data_folder=data_folder,
            corpus_config=self.config,
            compute_callback=compute_co_occurrence_callback,
            done_callback=self.display_explorer,
        )

        self.gui_load: load_gui.LoadGUI = load_gui.create_load_gui(
            data_folder=data_folder,
            filename_pattern=co_occurrence.FILENAME_PATTERN,
            loaded_callback=self.display_explorer,
        )

        self.gui_explore: explore_gui.ExploreGUI = None

    def layout(self):

        accordion = widgets.Accordion(children=[self.gui_load.layout(), self.gui_compute.layout()])

        accordion.set_title(0, "LOAD AN EXISTING CO-OCCURRENCE COMPUTATION")
        accordion.set_title(1, '...OR COMPUTE A NEW CO-OCCURRENCE')

        return widgets.VBox([accordion, view])

    @view.capture(clear_output=CLEAR_OUTPUT)
    def display_explorer(self, bundle: co_occurrence.Bundle, *_, **__):

        if bundle is None:
            return

        self.bundle = bundle
        self.trends_data = BundleTrendsData(bundle=bundle)
        self.gui_explore = explore_gui.ExploreGUI(bundle=bundle).setup().display(trends_data=self.trends_data)

        display(self.gui_explore.layout())
