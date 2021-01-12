from unittest.mock import Mock

import pytest
from penelope.notebook.dtm.to_DTM_gui import ComputeGUI, create_gui
from penelope.pipeline.config import CorpusConfig
from penelope.pipeline.pipelines import CorpusPipeline


def dummy_callback(*_, **__):
    pass


@pytest.mark.skip('')
def test_gui_create():

    gui = ComputeGUI()

    assert gui.layout() is not None


@pytest.mark.skip('')
def test_gui_setup():

    gui = ComputeGUI()
    config = Mock(spec=CorpusConfig)
    gui.setup(
        config=config,
        compute_callback=None,
        done_callback=dummy_callback,
    )

    assert gui.layout() is not None


@pytest.mark.skip('')
def test_create_gui():

    config = Mock(spec=CorpusConfig)
    gui: ComputeGUI = create_gui(
        corpus_folder='./tests/test_data',
        corpus_config=config,
        pipeline_factory=Mock(spec=CorpusPipeline),
        compute_callback=dummy_callback,
        done_callback=dummy_callback,
    )

    assert gui is not None
