import ipywidgets

from penelope.corpus import ExtractTaggedTokensOpts, TokensTransformOpts, VectorizeOpts
from penelope.notebook.gui_base import BaseGUI
from penelope.pipeline.config import CorpusConfig


def monkey_patch(*_, **__):
    ...


def test_gui_base_create_and_setup():  # pylint: disable=redefined-outer-name
    dummy_config = CorpusConfig.load(path='./tests/test_data/tranströmer/tranströmer.yml')
    gui = BaseGUI().setup(config=dummy_config, compute_callback=monkey_patch, done_callback=monkey_patch)
    assert isinstance(gui.transform_opts, TokensTransformOpts)
    assert isinstance(gui.extract_opts, ExtractTaggedTokensOpts)
    assert isinstance(gui.vectorize_opts, VectorizeOpts)
    layout = gui.layout()
    assert isinstance(layout, ipywidgets.VBox)
