from unittest.mock import Mock

import pytest
from penelope.notebook.dtm.compute_DTM_corpus import compute_document_term_matrix as compute_dtm
from penelope.notebook.interface import ComputeOpts
from penelope.pipeline.config import CorpusConfig


@pytest.fixture
def dummy_config():
    return CorpusConfig.load(path='./tests/test_data/ssi_corpus_config.yaml')


@pytest.mark.skip('')
def test_compute_dtm(dummy_config):  # pylint: disable=redefined-outer-name

    corpus = compute_dtm(dummy_config, pipeline_factory=None, args=Mock(spec=ComputeOpts))
    assert corpus is not None
