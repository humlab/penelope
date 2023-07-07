import IPython.display as ipy_display
import pandas as pd
from ipywidgets import Output, VBox

from penelope.corpus import dtm
from penelope.notebook import grid_utility as gu
from penelope.notebook.dtm import load_dtm_gui

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
    folder, tag = dtm.VectorizedCorpus.split(filename)
    corpus: dtm.VectorizedCorpus = dtm.load_corpus(
        folder=folder, tag=tag, tf_threshold=None, n_top=None, axis=None, group_by_year=False
    )
    mdw_gui = MDW_GUI().setup(corpus=corpus, done_callback=display_mdw)
    ipy_display.display(mdw_gui.layout())


def create_main_gui(folder: str, picked_callback=picked_corpus_handler) -> VBox:
    gui = load_dtm_gui.LoadGUI(folder=folder, pattern='*_vector_data.npz', picked_callback=picked_callback).setup()
    return VBox([gui.layout(), view_gui, view_display])
