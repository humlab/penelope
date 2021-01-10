import json

import yaml
from penelope.corpus.readers.interfaces import TextReaderOpts
from penelope.pipeline import CorpusConfig
from penelope.pipeline.interfaces import PipelinePayload

TEST_CONFIG = CorpusConfig.loads(
    """
corpus_name: ssi_unesco
corpus_pattern: '*.zip'
corpus_type: 1
language: english
pipeline_payload:
  document_index_sep: ;
  document_index_source: legal_instrument_index.csv
  filenames: null
  memory_store:
    lang: en
    lemma_column: lemma_
    pos_column: pos_
    spacy_model: en_core_web_sm
    tagger: spaCy
    text_column: text
  pos_schema_name: Universal
  source: legal_instrument_corpus.zip
tagged_tokens_filter_opts:
  data:
    is_alpha: null
    is_digit: null
    is_punct: false
    is_stop: null
text_reader_opts:
  as_binary: false
  filename_fields:
  - unesco_id:_:2
  - year:_:3
  - city:\\w+\\_\\d+\\_\\d+\\_\\d+\\_(.*)\\.txt
  filename_filter: null
  filename_pattern: '*.txt'
  index_field: null
"""
)


def test_json_dumps_and_loads_of_corpus_config_succeeds():

    first_dump_str = json.dumps(TEST_CONFIG, default=vars)
    deserialized_config = json.loads(first_dump_str)
    second_dump_str = json.dumps(deserialized_config, default=vars)

    assert first_dump_str == second_dump_str


def test_yaml_dumps_and_loads_of_corpus_config_succeeds():

    json_dump_str = json.dumps(TEST_CONFIG, default=vars)
    yaml_dump_str = yaml.dump(json_dump_str)

    _ = yaml.load(yaml_dump_str, Loader=yaml.FullLoader)


def test_dump_and_load_of_corpus_config_succeeds():

    dump_filename = './tests/output/corpus_config_test.yaml'
    TEST_CONFIG.dump(dump_filename)
    deserialized_config = CorpusConfig.load(dump_filename)

    assert json.dumps(TEST_CONFIG, default=vars) == json.dumps(deserialized_config, default=vars)


def test_find_config():

    c = CorpusConfig.find("ssi_corpus_config.yaml", './tests/test_data')
    assert c is not None


def test_corpus_config_set_folder():

    config: CorpusConfig = CorpusConfig(
        text_reader_opts=TextReaderOpts(),
        pipeline_payload=PipelinePayload(
            source="corpus.zip",
            document_index_source="document_index.csv",
        ),
    ).folder('/data')

    assert config.pipeline_payload.source == '/data/corpus.zip'
    assert config.pipeline_payload.document_index_source == '/data/document_index.csv'
