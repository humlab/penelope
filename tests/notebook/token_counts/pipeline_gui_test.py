from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from penelope.notebook.token_counts import pipeline_gui
from penelope.pipeline import config as corpus_config

# pylint: disable=redefined-outer-name


def fake_config() -> corpus_config.CorpusConfig:

    config: corpus_config.CorpusConfig = corpus_config.CorpusConfig.load('./tests/test_data/riksdagens-protokoll.yml')

    config.pipeline_payload.source = './tests/test_data/riksdagens-protokoll.1920-2019.9files.sparv4.csv.zip'
    config.pipeline_payload.document_index_source = None

    return config


@pytest.fixture(scope='module')
def config():
    return fake_config()


def monkey_patch(*_, **__):
    pass


@patch('penelope.notebook.token_counts.pipeline_gui.compute_token_count_data', monkey_patch)
@patch('penelope.notebook.token_counts.pipeline_gui.load_document_index', monkey_patch)
def test_create_token_count_gui_succeeds():

    corpus_folder: str = './tests/test_data'
    resources_folder: str = './tests/test_data'

    gui: pipeline_gui.TokenCountsGUI = pipeline_gui.create_token_count_gui(corpus_folder, resources_folder)

    assert gui is not None


def test_create_token_count_gui_create_succeeds():

    load_corpus_config_callback_is_called = False

    def load_corpus_config_callback(*_) -> corpus_config.CorpusConfig:
        nonlocal load_corpus_config_callback_is_called
        load_corpus_config_callback_is_called = True
        return fake_config()

    def load_document_index_callback(_: corpus_config.CorpusConfig) -> pd.DataFrame:
        return MagicMock(spec=pd.DataFrame)

    resources_folder: str = './tests/test_data'
    gui: pipeline_gui.TokenCountsGUI = (
        pipeline_gui.TokenCountsGUI(
            compute_callback=monkey_patch,
            load_document_index_callback=load_document_index_callback,
            load_corpus_config_callback=load_corpus_config_callback,
        )
        .setup(corpus_config.CorpusConfig.list(resources_folder))
        .display()
    )

    assert gui is not None
    assert load_corpus_config_callback_is_called


@pytest.mark.long_running
def test_load_document_index(config: corpus_config.CorpusConfig):

    config.pipeline_payload.source = 'riksdagens-protokoll.1920-2019.9files.sparv4.csv.zip'
    config.pipeline_payload.folders('./tests/test_data')
    config.checkpoint_opts.feather_folder = None

    document_index: pd.DataFrame = pipeline_gui.load_document_index(config)

    assert document_index is not None
