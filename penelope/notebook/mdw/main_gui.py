import IPython.display as ipy_display
import pandas as pd
from ipywidgets import Output, VBox

from penelope.corpus import dtm

from .. import grid_utility as gu
from .. import pick_file_gui
from .mdw_gui import MDW_GUI

view_display, view_gui = Output(), Output()


@view_display.capture(clear_output=True)
def display_mdw(corpus: dtm.VectorizedCorpus, df_mdw: pd.DataFrame):  # pylint: disable=unused-argument
    g = gu.table_widget(df_mdw)
    ipy_display.display(g)


@view_gui.capture(clear_output=True)
def picked_corpus_handler(filename: str):
    if not dtm.VectorizedCorpus.is_dump(filename):
        raise FileNotFoundError(f"Expected a DTM file, got {filename or 'None'}")
    corpus: dtm.VectorizedCorpus = dtm.VectorizedCorpus.load(filename=filename)
    mdw_gui = MDW_GUI().setup(corpus=corpus, done_callback=display_mdw)
    ipy_display.display(mdw_gui.layout())


def create_main_gui(folder: str, picked_callback=picked_corpus_handler) -> VBox:
    gui = pick_file_gui.PickFileGUI(folder=folder, pattern='*_vector_data.npz', picked_callback=picked_callback).setup()
    return VBox([gui.layout(), view_gui, view_display])
