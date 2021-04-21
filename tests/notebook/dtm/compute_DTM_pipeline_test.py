# from unittest.mock import Mock

import pytest

# from penelope.notebook.interface import ComputeOpts
from penelope.pipeline.config import CorpusConfig


@pytest.fixture
def dummy_config():
    return CorpusConfig.load(path='./tests/test_data/ssi_corpus_config.yml')


def test_compute_DTM():
    pass


def test_resolve_DTM_pipeline():
    pass


# @pytest.mark.skip('')
# def test_compute_dtm(dummy_config):  # pylint: disable=redefined-outer-name

#     corpus = compute_document_term_matrix(args=Mock(spec=ComputeOpts), corpus_config=dummy_config, pipeline_factory=None, )
#     assert corpus is not None
