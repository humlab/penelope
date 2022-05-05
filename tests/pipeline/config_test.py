import json
import os

import pytest
import yaml

from penelope.pipeline import CorpusConfig
from penelope.pipeline.interfaces import PipelinePayload

# pylint: disable=redefined-outer-name


@pytest.fixture
def corpus_config() -> CorpusConfig:
    return CorpusConfig.load('./tests/test_data/SSI.yml')


def test_json_dumps_and_loads_of_corpus_config_succeeds(corpus_config: CorpusConfig):

    first_dump_str = json.dumps(corpus_config, default=vars)
    deserialized_config = json.loads(first_dump_str)
    second_dump_str = json.dumps(deserialized_config, default=vars)

    assert first_dump_str == second_dump_str


def test_yaml_dumps_and_loads_of_corpus_config_succeeds(corpus_config: CorpusConfig):

    json_dump_str = json.dumps(corpus_config, default=vars)
    yaml_dump_str = yaml.dump(json_dump_str)

    _ = yaml.load(yaml_dump_str, Loader=yaml.FullLoader)


def test_dump_and_load_of_corpus_config_succeeds(corpus_config: CorpusConfig):

    os.makedirs('./tests/output', exist_ok=True)
    dump_filename = './tests/output/corpus_config_test.yml'
    corpus_config.dump(dump_filename)
    deserialized_config = CorpusConfig.load(dump_filename)

    # assert json.dumps(corpus_config, default=vars) == json.dumps(deserialized_config, default=vars)
    assert corpus_config.props == deserialized_config.props


def test_find_config(corpus_config: CorpusConfig):

    c = CorpusConfig.find("SSI.yml", './tests/test_data')

    assert json.dumps(c, default=vars) == json.dumps(corpus_config, default=vars)


def test_text_transform_opts(corpus_config: CorpusConfig):

    assert corpus_config.text_transform_opts is not None
    assert corpus_config.text_transform_opts.fix_hyphenation
    assert corpus_config.text_transform_opts.fix_whitespaces


def test_corpus_config_set_folders():

    payload = PipelinePayload(
        source="corpus.zip",
        document_index_source="document_index.csv",
    )
    payload.files("corpus.zip", "document_index.csv").folders('/data', method="replace")

    assert payload.source == '/data/corpus.zip'
    assert payload.document_index_source == '/data/document_index.csv'

    payload.files("/tmp/corpus.zip", "/tmp/document_index.csv").folders('/data', method="replace")

    assert payload.source == '/data/corpus.zip'
    assert payload.document_index_source == '/data/document_index.csv'

    payload.files("apa/corpus.zip", "apa/document_index.csv").folders('/data', method="join")

    assert payload.source == '/data/apa/corpus.zip'
    assert payload.document_index_source == '/data/apa/document_index.csv'
