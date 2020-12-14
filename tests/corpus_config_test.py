import json

import yaml
from penelope.corpus.readers import TaggedTokensFilterOpts, TextReaderOpts
from penelope.pipeline import CorpusConfig, CorpusType, PipelinePayload

TEST_CONFIG = CorpusConfig(
    corpus_name='ssi_unesco',
    corpus_type=CorpusType.Text,
    corpus_pattern="*.zip",
    language='english',
    text_reader_opts=TextReaderOpts(
        filename_fields=["unesco_id:_:2", "year:_:3", r'city:\w+\_\d+\_\d+\_\d+\_(.*)\.txt'],
        index_field=None,  # Use filename as key
        filename_filter=None,
        filename_pattern="*.txt",
        as_binary=False,
    ),
    tagged_tokens_filter_opts=TaggedTokensFilterOpts(
        is_alpha=None,
        is_space=False,
        is_punct=False,
        is_digit=None,
        is_stop=None,
    ),
    pipeline_payload=PipelinePayload(
        source="legal_instrument_corpus.zip",
        document_index_source="legal_instrument_index.csv",
        document_index_key=None,
        document_index_sep=';',
        pos_schema_name="Universal",
        memory_store={
            'tagger': 'spaCy',
            'text_column': 'text',
            'pos_column': 'pos_',
            'lemma_column': 'lemma_',
            'spacy_model': "en_core_web_sm",
            'nlp': None,
            'lang': 'en',
        },
    ),
)


def test_json_dumps_and_loads_of_corpus_config_succeeds():

    first_dump_str = json.dumps(TEST_CONFIG, default=vars)
    deserialized_config = json.loads(first_dump_str)
    second_dump_str = json.dumps(deserialized_config, default=vars)

    assert first_dump_str == second_dump_str


def test_yaml_dumps_and_loads_of_corpus_config_succeeds():

    json_dump_str = json.dumps(TEST_CONFIG, default=vars)
    yaml_dump_str = yaml.dump(json_dump_str)

    _ = yaml.load(yaml_dump_str)


def test_dump_and_load_of_corpus_config_succeeds():

    dump_filename = './tests/output/corpus_config_test.yaml'
    TEST_CONFIG.dump(dump_filename)
    deserialized_config = CorpusConfig.load(dump_filename)

    assert json.dumps(TEST_CONFIG, default=vars) == json.dumps(deserialized_config, default=vars)
