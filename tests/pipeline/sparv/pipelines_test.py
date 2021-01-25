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

    """Loads a teagged data frame"""
    p: pipeline.CorpusPipeline = pipelines.to_tagged_frame_pipeline(config)

    p.exhaust()

    assert p.payload.document_index is not None
    assert 'year' in p.payload.document_index.columns
