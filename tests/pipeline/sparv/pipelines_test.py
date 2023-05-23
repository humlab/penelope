import inspect
import os
import pathlib

import pytest

from penelope import pipeline
from penelope.pipeline.sparv import pipelines

# pylint: disable=redefined-outer-name,non-ascii-name


def fake_config() -> pipeline.CorpusConfig:
    corpus_config: pipeline.CorpusConfig = pipeline.CorpusConfig.load('./tests/test_data/riksdagens-protokoll.yml')

    corpus_config.pipeline_payload.source = './tests/test_data/riksdagens-protokoll.test.sparv4.csv.zip'
    corpus_config.pipeline_payload.document_index_source = None

    return corpus_config


@pytest.fixture(scope='module')
def config():
    return fake_config()


def test_to_tagged_frame_pipeline(config: pipeline.CorpusConfig):
    """Loads a tagged data frame"""
    p: pipeline.CorpusPipeline = pipelines.to_tagged_frame_pipeline(
        corpus_config=config,
        corpus_source=config.pipeline_payload.source,
        enable_checkpoint=False,
        force_checkpoint=False,
    )

    p.exhaust()

    assert p.payload.document_index is not None
    assert 'year' in p.payload.document_index.columns


@pytest.mark.long_running
def test_to_tagged_frame_pipeline_checkpoint_tranströmer():
    config_filename = './tests/test_data/tranströmer.yml'
    source_filename = './tests/test_data/tranströmer_corpus_export.sparv4.csv.zip'
    tagged_corpus_source = f'./tests/output/{inspect.currentframe().f_code.co_name}.tagged_frame.zip'
    corpus_config: pipeline.CorpusConfig = pipeline.CorpusConfig.load(config_filename)

    corpus_config.pipeline_payload.source = source_filename
    corpus_config.pipeline_payload.document_index_source = None

    pathlib.Path(tagged_corpus_source).unlink(missing_ok=True)

    p: pipeline.CorpusPipeline = pipelines.to_tagged_frame_pipeline(
        corpus_config=corpus_config,
        corpus_source=source_filename,
        enable_checkpoint=False,
        force_checkpoint=False,
    ).checkpoint(tagged_corpus_source)

    p.exhaust()

    assert p.payload.document_index is not None
    assert 'n_raw_tokens' in p.payload.document_index.columns
    assert 'year' in p.payload.document_index.columns

    assert os.path.isfile(tagged_corpus_source)

    pathlib.Path(tagged_corpus_source).unlink(missing_ok=True)


@pytest.mark.long_running
def test_to_tagged_frame_pipeline_checkpoint_adds_token_counts():
    config_filename = './tests/test_data/tranströmer.yml'
    source_filename = './tests/test_data/tranströmer_corpus_export.sparv4.csv.zip'
    corpus_config: pipeline.CorpusConfig = pipeline.CorpusConfig.load(config_filename)

    corpus_config.pipeline_payload.source = source_filename
    corpus_config.pipeline_payload.document_index_source = None

    p: pipeline.CorpusPipeline = pipelines.to_tagged_frame_pipeline(
        corpus_config=corpus_config,
        corpus_source=source_filename,
        enable_checkpoint=True,
        force_checkpoint=True,
    )

    p.exhaust()

    assert p.payload.document_index is not None
    assert 'n_raw_tokens' in p.payload.document_index.columns
    assert 'year' in p.payload.document_index.columns
