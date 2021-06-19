# from unittest.mock import Mock

import pytest

# from penelope.notebook.interface import ComputeOpts
from penelope.pipeline.config import CorpusConfig


@pytest.fixture
def dummy_config():
    return CorpusConfig.load(path='./tests/test_data/SSI.yml')


def test_compute_DTM():
    pass


def test_resolve_DTM_pipeline():
    pass
