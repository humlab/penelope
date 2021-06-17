import inspect
import os
import pathlib

import pytest
from penelope import pipeline
from penelope.pipeline.sparv import pipelines

# pylint: disable=redefined-outer-name


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
        corpus_filename=config.pipeline_payload.source,
        enable_checkpoint=False,
        force_checkpoint=False,
    )

    p.exhaust()

    assert p.payload.document_index is not None
    assert 'year' in p.payload.document_index.columns


def test_to_tagged_frame_pipeline_checkpoint_tranströmer():
    config_filename = './tests/test_data/tranströmer.yml'
    source_filename = './tests/test_data/tranströmer_corpus_export.sparv4.csv.zip'
    checkpoint_filename = f'./tests/output/{inspect.currentframe().f_code.co_name}.checkpoint.zip'
    corpus_config: pipeline.CorpusConfig = pipeline.CorpusConfig.load(config_filename)

    corpus_config.pipeline_payload.source = source_filename
    corpus_config.pipeline_payload.document_index_source = None

    pathlib.Path(checkpoint_filename).unlink(missing_ok=True)

    p: pipeline.CorpusPipeline = pipelines.to_tagged_frame_pipeline(
        corpus_config=corpus_config,
        corpus_filename=source_filename,
        enable_checkpoint=True,
        force_checkpoint=False,
    ).checkpoint(checkpoint_filename)

    for _ in p.resolve():
        assert 'n_raw_tokens' in p.payload.document_index.columns

    assert os.path.isfile(checkpoint_filename)

    assert p.payload.document_index is not None
    assert 'year' in p.payload.document_index.columns

    pathlib.Path(checkpoint_filename).unlink(missing_ok=True)


def test_to_tagged_frame_pipeline_checkpoint_adds_token_counts():
    config_filename = './tests/test_data/tranströmer.yml'
    source_filename = './tests/test_data/tranströmer_corpus_export.sparv4.csv.zip'
    corpus_config: pipeline.CorpusConfig = pipeline.CorpusConfig.load(config_filename)

    corpus_config.pipeline_payload.source = source_filename
    corpus_config.pipeline_payload.document_index_source = None

    p: pipeline.CorpusPipeline = pipelines.to_tagged_frame_pipeline(
        corpus_config=corpus_config,
        corpus_filename=source_filename,
        enable_checkpoint=True,
        force_checkpoint=True,
    )

    for _ in p.resolve():
        assert 'n_raw_tokens' in p.payload.document_index.columns

    assert p.payload.document_index is not None
    assert 'year' in p.payload.document_index.columns


@pytest.mark.skip(reason="NotImplemented")
def test_to_numeric_tagged_frame_pipeline():

    config_filename = './tests/test_data/tranströmer.yml'
    checkpoint_filename = f'./tests/output/{inspect.currentframe().f_code.co_name}.checkpoint.zip'

    corpus_config: pipeline.CorpusConfig = pipeline.CorpusConfig.load(config_filename)

    pathlib.Path(checkpoint_filename).unlink(missing_ok=True)

    p: pipeline.CorpusPipeline = pipeline.CorpusPipeline(config=corpus_config, force_checkpoint=True).checkpoint(
        checkpoint_filename
    )

    document_tuples = p.to_document_content_tuple().to_list()

    assert document_tuples is not None

    raise NotImplementedError()
