from unittest.mock import patch

import pandas as pd
import pytest

from penelope.notebook.token_counts import pipeline_gui
from penelope.pipeline import config as corpus_config

# pylint: disable=redefined-outer-name


def fake_config() -> corpus_config.CorpusConfig:
    config: corpus_config.CorpusConfig = corpus_config.CorpusConfig.load(
        './tests/test_data/riksdagens_protokoll/riksdagens-protokoll.yml'
    )

    config.pipeline_payload.source = (
        './tests/test_data/riksdagens_protokoll/riksdagens-protokoll.1920-2019.9files.sparv4.csv.zip'
    )
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
    resources_folder: str = './tests/test_data'

    config_filenames = corpus_config.CorpusConfig.list_all(resources_folder, recursive=True, try_load=True)

    gui: pipeline_gui.TokenCountsGUI = pipeline_gui.TokenCountsGUI()

    gui = gui.setup(config_filenames=config_filenames)

    assert gui is not None


@patch('penelope.notebook.token_counts.pipeline_gui.load_document_index', monkey_patch)
def test_create_token_count_gui_display_succeeds():
    resources_folder: str = './tests/test_data'
    gui: pipeline_gui.TokenCountsGUI = pipeline_gui.TokenCountsGUI(compute_callback=monkey_patch).setup(
        corpus_config.CorpusConfig.list_all(resources_folder)
    )

    assert gui.layout() is not None
    assert gui.display() is not None


@pytest.mark.long_running
def test_load_document_index(config: corpus_config.CorpusConfig):
    config.pipeline_payload.source = 'riksdagens-protokoll.1920-2019.9files.sparv4.csv.zip'
    config.pipeline_payload.folders('./tests/test_data/riksdagens_protokoll')
    config.serialize_opts.feather_folder = None

    document_index: pd.DataFrame = pipeline_gui.load_document_index(config)

    assert document_index is not None
