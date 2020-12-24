from dataclasses import dataclass
from typing import Callable

import ipywidgets as widgets
from penelope.corpus import VectorizedCorpus
from penelope.pipeline import CorpusConfig, CorpusPipeline
from penelope.utility import get_logger

from ..gui_base import BaseGUI

logger = get_logger('penelope')


@dataclass
class ComputeGUI(BaseGUI):
    def layout(self, hide_input=False, hide_output=False):
        layout = super().layout(hide_input, hide_output)
        return layout

    def setup(self, *, config: CorpusConfig, compute_callback: Callable):
        super().setup(config=config, compute_callback=compute_callback)
        return self


def create_gui(
    *,
    corpus_folder: str,
    corpus_config: CorpusConfig,
    pipeline_factory: Callable[[], CorpusPipeline],
    done_callback: Callable[[CorpusPipeline, VectorizedCorpus, str, str, widgets.Output], None],
    compute_document_term_matrix: Callable,
) -> ComputeGUI:
    """Returns a GUI for turning a corpus pipeline to a document-term-matrix (DTM)"""
    corpus_config.folder(corpus_folder)
    gui = ComputeGUI(
        default_corpus_path=corpus_folder,
        default_corpus_filename=(corpus_config.pipeline_payload.source or ''),
        default_target_folder=corpus_folder,
    ).setup(
        config=corpus_config,
        compute_callback=lambda g: compute_document_term_matrix(
            corpus_config=corpus_config,
            pipeline_factory=pipeline_factory,
            args=g,
            done_callback=done_callback,
            persist=True,
        ),
    )

    return gui
