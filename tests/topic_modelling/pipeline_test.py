import uuid
from typing import List
from unittest.mock import MagicMock, Mock

import pytest
from penelope.corpus.readers import ExtractTaggedTokensOpts, TextReaderOpts, TextTransformOpts
from penelope.pipeline import ContentType, CorpusConfig, CorpusPipeline, DocumentPayload, ITask
from penelope.pipeline.topic_model.tasks import ToTopicModel
from penelope.utility import PropertyValueMaskingOpts
from tests.fixtures import TranströmerCorpus
from tests.pipeline.fixtures import SPARV_TAGGED_COLUMNS

# pylint: disable=redefined-outer-name


def fake_config() -> CorpusConfig:
    corpus_config: CorpusConfig = CorpusConfig.load('./tests/test_data/SSI.yml')
    corpus_config.pipeline_payload.source = './tests/test_data/legal_instrument_five_docs_test.zip'
    corpus_config.pipeline_payload.document_index_source = './tests/test_data/legal_instrument_five_docs_test.csv'
    return corpus_config


DEFAULT_ENGINE_ARGS = {
    'n_topics': 4,
    'passes': 1,
    'random_seed': 42,
    'alpha': 'auto',
    'workers': 1,
    'max_iter': 100,
    'prefix': '',
}


@pytest.fixture(scope='module')
def config():
    return fake_config()


def test_topic_model_using_pipeline(config: CorpusConfig, en_nlp):
    target_name: str = f'{uuid.uuid1()}'
    transform_opts: TextTransformOpts = TextTransformOpts()
    reader_opts: TextReaderOpts = TextReaderOpts(filename_pattern="*.txt", filename_fields="year:_:1")
    attributes: List[str] = ['text', 'lemma_', 'pos_']
    extract_opts: ExtractTaggedTokensOpts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes='|VERB|NOUN|', pos_paddings='|ADJ|', **config.pipeline_payload.tagged_columns_names
    )
    transform_opts = None
    filter_opts: PropertyValueMaskingOpts = PropertyValueMaskingOpts(is_punct=False)
    pipeline: CorpusPipeline = (
        CorpusPipeline(config=config)
        .load_text(reader_opts=reader_opts, transform_opts=TextTransformOpts())
        .set_spacy_model(en_nlp)
        .text_to_spacy()
        .spacy_to_tagged_frame(attributes=attributes)
        # .checkpoint(f'./tests/output/{uuid.uuid1()}.zip')
        .tagged_frame_to_tokens(extract_opts=extract_opts, filter_opts=filter_opts, transform_opts=transform_opts)
        .to_topic_model(
            corpus_filename=None,
            target_folder="./tests/output",
            target_name=target_name,
            engine="gensim_lda-multicore",
            engine_args=DEFAULT_ENGINE_ARGS,
            store_corpus=True,
            store_compressed=True,
        )
    ).exhaust()
    assert pipeline is not None


def test_topic_model_task_with_token_stream_and_document_index():

    corpus = TranströmerCorpus()

    payload_stream = lambda: [
        DocumentPayload(content_type=ContentType.TOKENS, filename=filename, content=tokens)
        for filename, tokens in corpus
    ]

    pipeline = Mock(
        spec=CorpusPipeline,
        **{'payload.memory_store': SPARV_TAGGED_COLUMNS, 'payload.document_index': corpus.document_index},
    )

    prior = MagicMock(spec=ITask, outstream=payload_stream, out_content_type=ContentType.TOKENS)

    task: ToTopicModel = ToTopicModel(
        pipeline=pipeline,
        prior=prior,
        corpus_filename=None,
        target_folder="./tests/output",
        target_name="APA",
        engine="gensim_lda-multicore",
        engine_args=DEFAULT_ENGINE_ARGS,
        store_corpus=True,
        store_compressed=True,
    )

    task.setup()
    task.enter()
    payload: DocumentPayload = next(task.process_stream())

    assert payload is not None
