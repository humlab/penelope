from typing import Any
from unittest.mock import Mock, patch

import penelope.notebook.co_occurrence.to_co_occurrence_gui as to_co_occurrence_gui
import penelope.notebook.utility as notebook_utility
from penelope.notebook import interface
from penelope.pipeline.config import CorpusConfig


def dummy_config():
    return CorpusConfig.load(path='./tests/test_data/ssi_corpus_config.yml')


@patch('penelope.notebook.utility.FileChooserExt2', Mock(spec=notebook_utility.FileChooserExt2))
def test_to_co_occurrence_create_gui(z):  # pylint: disable=unused-argument
    def done_callback(_: Any, __: interface.ComputeOpts):  # pylint: disable=unused-argument
        pass

    def compute_callback(args, corpus_config):  # pylint: disable=unused-argument
        pass

    corpus_config = dummy_config()

    gui = to_co_occurrence_gui.create_compute_gui(
        corpus_folder='./tests/test_data',
        data_folder='./tests/test_data',
        corpus_config=corpus_config,
        compute_callback=compute_callback,
        done_callback=done_callback,
    )

    assert gui is not None


@patch('penelope.notebook.utility.FileChooserExt2', Mock(spec=notebook_utility.FileChooserExt2))
def test_GUI_setup(z):  # pylint: disable=unused-argument
    def done_callback(*_, **__):
        pass

    def compute_callback(args, corpus_config):  # pylint: disable=unused-argument
        pass

    corpus_config = dummy_config()
    gui = to_co_occurrence_gui.ComputeGUI(
        default_corpus_path='./tests/test_data',
        default_corpus_filename='',
        default_data_folder='./tests/output',
    ).setup(
        config=corpus_config,
        compute_callback=compute_callback,
        done_callback=done_callback,
    )

    # layout = gui.layout()
    # gui._compute_handler({})  # pylint: disable=protected-access

    assert gui is not None
