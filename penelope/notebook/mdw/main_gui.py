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
def default_loaded_callback(folder: str, tag: str):
    corpus: dtm.VectorizedCorpus = dtm.load_corpus(
        folder=folder, tag=tag, tf_threshold=None, n_top=None, axis=None, group_by_year=False
    )
    mdw_gui = MDW_GUI().setup(corpus=corpus, done_callback=display_mdw)
    ipy_display.display(mdw_gui.layout())


def create_main_gui(corpus_folder: str, loaded_callback=default_loaded_callback) -> VBox:
    gui = load_dtm_gui.LoadGUI(folder=corpus_folder, done_callback=loaded_callback).setup()
    return VBox([gui.layout(), view_gui, view_display])
