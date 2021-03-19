import os

import pytest
from penelope import pipeline
from penelope.pipeline.sparv import pipelines

# pylint: disable=redefined-outer-name


def fake_config() -> pipeline.CorpusConfig:
    corpus_config: pipeline.CorpusConfig = pipeline.CorpusConfig.load(
        './tests/test_data/riksdagens-protokoll.yml'
    ).files(
        source='./tests/test_data/riksdagens-protokoll.test.sparv4.csv.zip',
        index_source=None,
    )
    return corpus_config


@pytest.fixture(scope='module')
def config():
    return fake_config()


def test_to_tagged_frame_pipeline(config: pipeline.CorpusConfig):

    """Loads a tagged data frame"""
    p: pipeline.CorpusPipeline = pipelines.to_tagged_frame_pipeline(config)

    p.exhaust()

    assert p.payload.document_index is not None
    assert 'year' in p.payload.document_index.columns


def test_to_tagged_frame_pipeline_checkpoint_tranströmer():

    config_filename = './tests/test_data/tranströmer.yml'
    source_filename = './tests/test_data/tranströmer_corpus_export.sparv4.csv.zip'
    target_filename = './tests/output/tranströmer_corpus_pos_csv.zip'
    corpus_config: pipeline.CorpusConfig = pipeline.CorpusConfig.load(config_filename,).files(
        source=source_filename,
        index_source=None,
    )
    assert corpus_config is not None

    p: pipeline.CorpusPipeline = pipelines.to_tagged_frame_pipeline(corpus_config).checkpoint(target_filename)

    p.exhaust()

    assert os.path.isfile(target_filename)

    assert p.payload.document_index is not None
    assert 'year' in p.payload.document_index.columns


def test_to_numeric_tagged_frame_pipeline():

    config_filename = './tests/test_data/tranströmer.yml'
    checkpoint_filename = './tests/output/tranströmer_corpus_pos_csv.zip'

    corpus_config: pipeline.CorpusConfig = pipeline.CorpusConfig.load(config_filename)

    p: pipeline.CorpusPipeline = pipeline.CorpusPipeline(config=corpus_config).checkpoint(checkpoint_filename)

    document_tuples = p.to_document_content_tuple().to_list()

    assert document_tuples is not None

    raise NotImplementedError()

    # token2id = generate_token2id()

    # for _, document in document_tuples:
    #     document = document_tuples[0][1]

    # assert document is not None
