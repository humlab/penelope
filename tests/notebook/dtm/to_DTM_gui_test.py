import pytest
from penelope.pipeline.pipelines import CorpusPipeline
from unittest.mock import Mock
from penelope.pipeline.config import CorpusConfig
from penelope.notebook.dtm.to_DTM_gui import GUI, create_gui


def dummy_callback(*_,**__):
    pass

@pytest.mark.skip('')
def test_gui_create():

    gui = GUI()

    assert gui.layout() is not None

@pytest.mark.skip('')
def test_gui_setup():

    gui = GUI()
    config = Mock(spec=CorpusConfig)
    gui.setup(config=config, compute_callback=None)

    assert gui.layout() is not None

@pytest.mark.skip('')
def test_create_gui():

    config = Mock(spec=CorpusConfig)
    gui = create_gui(
        corpus_folder='./tests/test_data',
        corpus_config=config,
        compute_document_term_matrix=dummy_callback,
        pipeline_factory=Mock(spec=CorpusPipeline),
        done_callback=dummy_callback,
    )

    assert gui is not None
