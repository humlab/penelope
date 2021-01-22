import ipywidgets as widgets
import penelope.corpus.dtm as dtm
import penelope.notebook.dtm as dtm_gui
import penelope.notebook.word_trends as word_trends
import penelope.pipeline as pipeline
import penelope.workflows as workflows
from IPython.core.display import display
from penelope.corpus import VectorizedCorpus
from penelope.notebook import interface

view = widgets.Output(layout={'border': '2px solid green'})

LAST_ARGS = None
LAST_CORPUS_CONFIG = None
LAST_CORPUS = None


@view.capture(clear_output=True)
def loaded_callback(
    corpus: VectorizedCorpus,
    corpus_folder: str,
    corpus_tag: str,
):
    global LAST_CORPUS
    LAST_CORPUS = corpus
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
def computed_callback(
    corpus: VectorizedCorpus,
    opts: interface.ComputeOpts,
) -> None:

    loaded_callback(corpus=corpus, corpus_folder=opts.target_folder, corpus_tag=opts.corpus_tag)


@view.capture(clear_output=True)
def compute_callback(args: interface.ComputeOpts, corpus_config: pipeline.CorpusConfig) -> dtm.VectorizedCorpus:
    global LAST_ARGS, LAST_CORPUS_CONFIG
    LAST_ARGS = args
    LAST_CORPUS_CONFIG = corpus_config
    corpus: dtm.VectorizedCorpus = workflows.document_term_matrix.compute(args=args, corpus_config=corpus_config)
    return corpus


def create_to_dtm_gui(
    corpus_folder: str,
    corpus_config: str,
    resources_folder: str = None,
) -> widgets.CoreWidget:

    resources_folder = resources_folder or corpus_folder
    config: pipeline.CorpusConfig = pipeline.CorpusConfig.find(corpus_config, resources_folder).folder(corpus_folder)
    gui_compute: dtm_gui.ComputeGUI = dtm_gui.create_compute_gui(
        corpus_folder=corpus_folder,
        corpus_config=config,
        compute_callback=compute_callback,
        done_callback=computed_callback,
    )

    gui_load: dtm_gui.LoadGUI = dtm_gui.create_load_gui(
        corpus_folder=corpus_folder,
        loaded_callback=loaded_callback,
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
