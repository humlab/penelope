import ipywidgets
import pytest
from penelope.corpus import VectorizeOpts
from penelope.corpus.readers.interfaces import ExtractTaggedTokensOpts
from penelope.corpus.tokens_transformer import TokensTransformOpts
from penelope.notebook.gui_base import BaseGUI
from penelope.pipeline.config import CorpusConfig
from penelope.utility import PropertyValueMaskingOpts


def monkey_patch(*_, **__):
    ...


def test_gui_base_create_and_payout():
    gui = BaseGUI()
    layout = gui.layout()
    assert isinstance(layout, ipywidgets.VBox)


@pytest.mark.fixture
def dummy_config():
    return CorpusConfig.load(path='./tests/test_data/SSI.yml')


def test_gui_base_create_and_setup(dummy_config):  # pylint: disable=redefined-outer-name
    gui = BaseGUI().setup(config=dummy_config, compute_callback=monkey_patch, done_callback=monkey_patch)
    assert isinstance(gui.filter_opts, PropertyValueMaskingOpts)
    assert isinstance(gui.transform_opts, TokensTransformOpts)
    assert isinstance(gui.extract_opts, ExtractTaggedTokensOpts)
    assert isinstance(gui.vectorize_opts, VectorizeOpts)
