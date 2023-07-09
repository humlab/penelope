from penelope.pipeline import CorpusConfig

from ..gui_base import BaseGUI, ComputeCallback, DoneCallback


class ComputeGUI(BaseGUI):
    def layout(self, hide_input=False, hide_output=False):
        layout = super().layout(hide_input, hide_output)
        return layout

    def setup(self, *, config: CorpusConfig, compute_callback: ComputeCallback, done_callback: DoneCallback):
        super().setup(config=config, compute_callback=compute_callback, done_callback=done_callback)
        return self


def create_compute_gui(
    *,
    corpus_folder: str,
    data_folder: str,
    compute_callback: ComputeCallback,
    done_callback: DoneCallback,
    config: str | CorpusConfig = None,
) -> ComputeGUI:
    """Returns a GUI for turning a corpus pipeline to a document-term-matrix (DTM)"""

    if config is not None:
        config: CorpusConfig = CorpusConfig.find(config, corpus_folder).folders(corpus_folder)

    gui = ComputeGUI(
        default_corpus_path=corpus_folder,
        default_corpus_filename='',
        default_data_folder=data_folder,
    ).setup(
        config=config,
        compute_callback=compute_callback,
        done_callback=done_callback,
    )

    return gui
