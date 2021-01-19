from dataclasses import dataclass
from typing import Callable, Optional, Union

import ipywidgets as widgets
import penelope.notebook.co_occurrence as co_occurrence_gui
from IPython.core.display import display
from penelope import co_occurrence, pipeline, workflows
from penelope.notebook.interface import ComputeOpts
from penelope.notebook.word_trends.trends_data import TrendsData

view = widgets.Output(layout={'border': '2px solid green'})


def create(
    data_folder: str,
    filename_pattern: str = co_occurrence.CO_OCCURRENCE_FILENAME_PATTERN,
    loaded_callback: Callable[[co_occurrence.Bundle], None] = None,
) -> co_occurrence_gui.LoadGUI:

    gui: co_occurrence_gui.LoadGUI = co_occurrence_gui.LoadGUI(default_path=data_folder).setup(
        filename_pattern=filename_pattern, load_callback=co_occurrence.load_bundle, loaded_callback=loaded_callback
    )
    return gui


@view.capture(clear_output=True)
def compute_co_occurrence_callback(
    corpus_config: pipeline.CorpusConfig,
    args: ComputeOpts,
    checkpoint_file: Optional[str] = None,
) -> co_occurrence.Bundle:
    bundle = workflows.co_occurrence.compute(
        args=args,
        corpus_config=corpus_config,
        checkpoint_file=checkpoint_file,
    )
    return bundle


@dataclass
class MainGUI:
    def __init__(
        self,
        corpus_config: Union[pipeline.CorpusConfig, str],
        corpus_folder: str,
        resources_folder: str,
    ) -> widgets.VBox:

        self.trends_data: TrendsData = None
        self.config = (
            corpus_config
            if isinstance(corpus_config, pipeline.CorpusConfig)
            else pipeline.CorpusConfig.find(corpus_config, resources_folder).folder(corpus_folder)
        )

        self.gui_compute: co_occurrence_gui.ComputeGUI = co_occurrence_gui.create_compute_gui(
            corpus_folder=corpus_folder,
            corpus_config=self.config,
            compute_callback=compute_co_occurrence_callback,
            done_callback=self.display_explorer,
        )

        self.gui_load: co_occurrence_gui.LoadGUI = co_occurrence_gui.create_load_gui(
            data_folder=corpus_folder,
            filename_pattern=co_occurrence.CO_OCCURRENCE_FILENAME_PATTERN,
            loaded_callback=self.display_explorer,
        )

        self.gui_explore: co_occurrence_gui.ExploreGUI = None

    def layout(self):

        accordion = widgets.Accordion(
            children=[
                widgets.VBox(
                    [
                        self.gui_load.layout(),
                    ],
                    layout={'border': '1px solid black', 'padding': '16px', 'margin': '4px'},
                ),
                widgets.VBox(
                    [
                        self.gui_compute.layout(),
                    ],
                    layout={'border': '1px solid black', 'padding': '16px', 'margin': '4px'},
                ),
            ]
        )

        accordion.set_title(0, "LOAD AN EXISTING CO-OCCURRENCE COMPUTATION")
        accordion.set_title(1, '...OR COMPUTE A NEW CO-OCCURRENCE')
        # accordion.set_title(2, '...OR LOAD AND EXPLORE A CO-OCCURRENCE DTM')
        # accordion.set_title(3, '...OR COMPUTE OR DOWNLOAD CO-OCCURRENCES AS EXCEL')

        return widgets.VBox([accordion, view])

    @view.capture(clear_output=True)
    def display_explorer(self, bundle: co_occurrence.Bundle, *_, **__):

        self.trends_data = co_occurrence.to_trends_data(bundle).update()
        self.gui_explore = co_occurrence_gui.ExploreGUI().setup().display(trends_data=self.trends_data)

        display(self.gui_explore.layout())
