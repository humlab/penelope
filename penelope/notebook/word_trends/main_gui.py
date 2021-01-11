from functools import partial
from typing import Callable

import ipywidgets as widgets
import penelope.notebook.dtm.compute_DTM_corpus as compute_DTM_corpus
import penelope.notebook.dtm.compute_DTM_pipeline as compute_DTM_pipeline
import penelope.notebook.dtm.load_DTM_gui as load_DTM_gui
import penelope.notebook.dtm.to_DTM_gui as to_DTM_gui
import penelope.notebook.word_trends as word_trends
import penelope.pipeline as pipeline
from IPython.core.display import display
from penelope.corpus import VectorizedCorpus

view = widgets.Output(layout={'border': '2px solid green'})


def compute_DTM(corpus_type: pipeline.CorpusType) -> Callable:
    if corpus_type == pipeline.CorpusType.SparvCSV:
        return compute_DTM_corpus.compute_document_term_matrix
    if corpus_type == pipeline.CorpusType.SpacyCSV:
        return compute_DTM_pipeline.compute_document_term_matrix
    raise ValueError(f"Unsupported (not implemented) corpus type for DTM: {corpus_type} ")


@view.capture(clear_output=True)
def corpus_loaded_callback(
    corpus: VectorizedCorpus,
    corpus_tag: str,
    corpus_folder: str,
    **_,
):
    trends_data: word_trends.TrendsData = word_trends.TrendsData(
        corpus=corpus,
        corpus_folder=corpus_folder,
        corpus_tag=corpus_tag,
        n_count=25000,
    ).update()

    gui = word_trends.GofTrendsGUI(
        gofs_gui=word_trends.GoFsGUI().setup(),
        trends_gui=word_trends.TrendsGUI().setup(),
    )

    display(gui.layout())
    gui.display(trends_data=trends_data)


@view.capture(clear_output=True)
def compute_dtm(compute_dtm, *args, **kwargs):
    compute_dtm.compute_document_term_matrix(*args, **kwargs)


def create_to_dtm_gui(
    corpus_folder: str,
    corpus_config: str,
    resources_folder: str = None,
    dtm_pipeline: Callable = None
) -> widgets.CoreWidget:

    resources_folder = resources_folder or corpus_folder
    config: pipeline.CorpusConfig = pipeline.CorpusConfig.find(corpus_config, resources_folder).folder(
        corpus_folder
    )
    compute_dtm_fx = compute_DTM(config.corpus_type)
    gui_compute: to_DTM_gui.ComputeGUI = to_DTM_gui.create_gui(
        corpus_folder=corpus_folder,
        corpus_config=config,
        pipeline_factory=dtm_pipeline,
        compute_document_term_matrix=partial(compute_dtm, compute_dtm_fx),
        done_callback=corpus_loaded_callback,
    )

    gui_load: load_DTM_gui.LoadGUI = load_DTM_gui.create_gui(
        corpus_folder=corpus_folder,
        loaded_callback=corpus_loaded_callback,
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
