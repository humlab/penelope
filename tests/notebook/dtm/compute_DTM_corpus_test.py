from unittest.mock import Mock, patch

import pytest
from penelope.notebook.dtm.compute_DTM_corpus import compute_document_term_matrix
from penelope.notebook.dtm.to_DTM_gui import GUI
from penelope.pipeline.config import CorpusConfig


@pytest.mark.fixture
def dummy_config():
    return CorpusConfig.load(path='./tests/test_data/ssi_corpus_config.yaml')


@pytest.mark.skip('')
def test_compute_document_term_matrix(dummy_config):

    done_called = False

    def done_callback(*_, **__):
        nonlocal done_called
        done_called = True

    compute_document_term_matrix(
        dummy_config, pipeline_factory=None, args=Mock(spec=GUI), done_callback=done_callback, persist=False
    )
    assert done_called
