import IPython.display as ipy_display
from ipywidgets import Output, VBox

from penelope.corpus import dtm
from penelope.notebook import grid_utility as gu
from penelope.notebook.dtm import load_dtm_gui

from .mdw_gui import create_mdw_gui

view_display, view_gui = Output(), Output()


@view_display.capture(clear_output=True)
def display_mdw(corpus: dtm.VectorizedCorpus, df_mdw):  # pylint: disable=unused-argument
    g = gu.table_widget(df_mdw)
    ipy_display.display(g)


@view_gui.capture(clear_output=True)
def default_loaded_callback(
    corpus: dtm.VectorizedCorpus, corpus_folder: str, corpus_tag: str
):  # pylint: disable=unused-argument
    mdw_gui = create_mdw_gui(corpus, done_callback=display_mdw)
    ipy_display.display(mdw_gui.layout())


def create_main_gui(corpus_folder: str, loaded_callback=default_loaded_callback) -> VBox:

    gui = load_dtm_gui.create_load_gui(corpus_folder=corpus_folder, loaded_callback=loaded_callback)
    return VBox([gui.layout(), view_gui, view_display])
