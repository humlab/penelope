from typing import Any
import IPython.display as ipy_display
import pandas as pd
from ipywidgets import VBox, HBox

from penelope.corpus import dtm

from .. import grid_utility as gu
from .. import pick_file_gui as pfg
from .mdw_gui import MDW_GUI


def create_main_gui(folder: str) -> VBox:

    view: HBox = HBox()

    def display_mdw(result: pd.DataFrame, sender: pfg.PickFileGUI):  # pylint: disable=unused-argument
        g = gu.table_widget(result)
        # ipy_display.display(g)
        view.children = [g]

    def picked_corpus_handler(*, filename: str, sender: pfg.PickFileGUI):
        if not dtm.VectorizedCorpus.is_dump(filename):
            raise FileNotFoundError(f"Expected a DTM file, got {filename or 'None'}")
        gui.payload.corpus = dtm.VectorizedCorpus.load(filename=filename)

    gui: pfg.PickFileGUI = pfg.PickFileGUI(folder=folder, pattern='*_vector_data.npz', picked_callback=picked_corpus_handler).setup()
    gui.payload = MDW_GUI().setup(computed_callback=display_mdw).setup(computed_callback=display_mdw)
    gui.add(VBox([gui.payload.layout(), view]))

    return gui
