import ipywidgets
import pytest
from penelope.corpus.readers.interfaces import ExtractTaggedTokensOpts, TaggedTokensFilterOpts
from penelope.corpus.tokens_transformer import TokensTransformOpts
from penelope.corpus.vectorizer import VectorizeOpts
from penelope.notebook.gui_base import BaseGUI
from penelope.pipeline.config import CorpusConfig


def test_gui_base_create_and_payout():
    gui = BaseGUI()
    layout = gui.layout()
    assert isinstance(layout, ipywidgets.VBox)


@pytest.mark.fixture
def dummy_config():
    return CorpusConfig.load(path='./tests/test_data/ssi_corpus_config.yaml')


def test_gui_base_create_and_setup(dummy_config):
    gui = BaseGUI().setup(config=dummy_config, compute_callback=None)
    assert isinstance(gui.tagged_tokens_filter_opts, TaggedTokensFilterOpts)
    assert isinstance(gui.tokens_transform_opts, TokensTransformOpts)
    assert isinstance(gui.extract_tagged_tokens_opts, ExtractTaggedTokensOpts)
    assert isinstance(gui.vectorize_opts, VectorizeOpts)
