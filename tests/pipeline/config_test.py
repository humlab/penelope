import json
import os

import pytest
import yaml

from penelope.pipeline import CorpusConfig
from penelope.pipeline.config import DependencyResolver
from penelope.pipeline.interfaces import PipelinePayload

# pylint: disable=redefined-outer-name


@pytest.fixture
def corpus_config() -> CorpusConfig:
    return CorpusConfig.load('./tests/test_data/tranströmer/tranströmer.yml')


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
    c = CorpusConfig.find("tranströmer.yml", './tests/test_data')

    assert json.dumps(c, default=vars) == json.dumps(corpus_config, default=vars)


def test_find_all_configs():
    configs: list[CorpusConfig] = CorpusConfig.find_all('./tests/test_data/tranströmer')

    assert len(configs) > 0

    assert any(x for x in configs if x.corpus_name == "tranströmer")


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


def test_dependency_store(corpus_config: CorpusConfig):
    store: dict = corpus_config.dependency_store()
    assert store is not None

    assert set(store.keys()) == {'tagger', 'text_repository', 'render_text', 'text_loader', 'dummy_dependency'}


def test_resolve_dependencies():
    store: dict = {'dummy_dependency': {'class_name': 'list', 'arguments': ['apa']}}

    assert DependencyResolver.resolve_key('dummy_dependency', store) == ['a', 'p', 'a']

    store: dict = {
        'dummy_dependency': {'class_name': 'collections.Counter', 'arguments': [['a', 'p', 'a', 'b', 'c', 'a']]}
    }

    counter = DependencyResolver.resolve_key('dummy_dependency', store)

    assert counter['a'] == 3


TEST_YML_STR: str = """
  tagger:
    class_name: penelope.pipeline.stanza.StanzaTagger
    options:
  text_repository:
    class_name: penelope.corpus.render.TextRepository
    options:
      source: config@text_loader
      document_index: local@document_index
      transforms: normalize-whitespace
    dependencies:
      document_index:
        class_name: penelope.corpus.render.load_document_index
        options:
          filename: ./tests/test_data/tranströmer/tranströmer_corpus.csv
          sep: "\t"
  render_text:
    class_name: penelope.corpus.render.RenderService
    options:
      template: ./tests/test_data/tranströmer/tranströmer_corpus.jinja
      links_registry:
        PDF: '<a href="{{document_name}}.pdf">PDF</a>'
        MD: '<a href="{{document_name}}.txt">MD</a>'
  text_loader:
    class_name: penelope.corpus.render.ZippedTextCorpusLoader
    options:
      source: ./tests/test_data/tranströmer/tranströmer_corpus.zip
"""


def test_resolve_text_loader():
    store: dict = yaml.load(TEST_YML_STR, Loader=yaml.FullLoader)

    text_loader = DependencyResolver.resolve_key('text_loader', store)
    assert text_loader is not None


def test_resolve_text_repository():
    store: dict = yaml.load(TEST_YML_STR, Loader=yaml.FullLoader)

    text_repository = DependencyResolver.resolve_key('text_repository', store)
    assert text_repository is not None

    assert len(text_repository.filenames) > 0
    assert text_repository.get_text('tran_2020_02_test.txt').startswith('När året sparkar av sig')

    # assert text_repository.document_index == './tests/test_data/tranströmer/tranströmer_corpus.csv'


def test_resolve_render_text():
    store: dict = yaml.load(TEST_YML_STR, Loader=yaml.FullLoader)

    render_text = DependencyResolver.resolve_key('render_text', store)
    assert render_text is not None

    document_info: dict = {'document_name': "apa"}
    render_text.render(document_info, kind='text')

    assert document_info['text'] == 'apa'
