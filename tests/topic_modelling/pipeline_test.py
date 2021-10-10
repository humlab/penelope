import os
import uuid
from unittest.mock import MagicMock, Mock

import pytest
from penelope.corpus import ExtractTaggedTokensOpts, TokensTransformOpts
from penelope.pipeline import ContentType, CorpusConfig, CorpusPipeline, DocumentPayload, ITask
from penelope.pipeline.interfaces import ContentStream
from penelope.pipeline.topic_model.tasks import ToTopicModel
from penelope.utility import PropertyValueMaskingOpts
from tests.fixtures import TranströmerCorpus
from tests.pipeline.fixtures import SPARV_TAGGED_COLUMNS

from . import utility

# pylint: disable=redefined-outer-name


# @pytest.fixture(scope='module')
# def ssi_topic_model_payload(config: CorpusConfig, en_nlp) -> DocumentPayload:
#     return utility.ssi_topic_model_payload(config, en_nlp)


def tranströmer_topic_model_payload(method: str) -> DocumentPayload:
    transform_opts: TokensTransformOpts = TokensTransformOpts()
    filter_opts: PropertyValueMaskingOpts = PropertyValueMaskingOpts()
    extract_opts: ExtractTaggedTokensOpts = ExtractTaggedTokensOpts(
        lemmatize=True,
        pos_includes='',
        pos_excludes='MAD|MID|PAD',
        text_column='token',
        lemma_column='baseform',
        pos_column='pos',
    )
    config: CorpusConfig = CorpusConfig.load('./tests/test_data/tranströmer.yml')
    corpus_filename: str = './tests/test_data/tranströmer_corpus_pos_csv.zip'
    target_name: str = f'{uuid.uuid1()}'
    p: CorpusPipeline = (
        CorpusPipeline(config=config)
        .load_tagged_frame(
            filename=corpus_filename,
            checkpoint_opts=config.checkpoint_opts,
            extra_reader_opts=config.text_reader_opts,
        )
        .tagged_frame_to_tokens(extract_opts=extract_opts, filter_opts=filter_opts, transform_opts=transform_opts)
        .to_topic_model(
            corpus_filename=None,
            target_folder="./tests/output",
            target_name=target_name,
            engine=method,
            engine_args=utility.DEFAULT_ENGINE_ARGS,
            store_corpus=True,
            store_compressed=True,
        )
    )

    payload: DocumentPayload = p.single()

    return payload


@pytest.mark.long_running
@pytest.mark.parametrize('method', ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_predict_topics(method: str):

    payload: DocumentPayload = tranströmer_topic_model_payload(method=method)
    config: CorpusConfig = CorpusConfig.load('./tests/test_data/tranströmer.yml')
    corpus_filename: str = './tests/test_data/tranströmer_corpus_pos_csv.zip'

    target_folder: str = './tests/output'
    target_name: str = f'{uuid.uuid1()}'

    model_folder: str = payload.content.get("target_folder")
    model_name: str = payload.content.get("target_name")

    transform_opts = TokensTransformOpts()
    filter_opts = PropertyValueMaskingOpts()
    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True,
        pos_includes='',
        pos_excludes='MAD|MID|PAD',
        **config.checkpoint_opts.tagged_columns,
    )
    payload: DocumentPayload = (
        CorpusPipeline(config=config)
        .load_tagged_frame(
            filename=corpus_filename,
            checkpoint_opts=config.checkpoint_opts,
            extra_reader_opts=config.text_reader_opts,
        )
        .tagged_frame_to_tokens(extract_opts=extract_opts, filter_opts=filter_opts, transform_opts=transform_opts)
        .predict_topics(
            model_folder=model_folder,
            model_name=model_name,
            target_folder=target_folder,
            target_name=target_name,
        )
    ).single()

    assert payload is not None


@pytest.mark.long_running
@pytest.mark.parametrize("method", ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_topic_model_task_with_token_stream_and_document_index(method):

    target_name: str = f'{uuid.uuid1()}'
    corpus = TranströmerCorpus()

    payload_stream = lambda: [
        DocumentPayload(content_type=ContentType.TOKENS, filename=filename, content=tokens)
        for filename, tokens in corpus
    ]

    pipeline = Mock(
        spec=CorpusPipeline,
        **{'payload.memory_store': SPARV_TAGGED_COLUMNS, 'payload.document_index': corpus.document_index},
    )

    prior = MagicMock(
        spec=ITask,
        outstream=payload_stream,
        content_stream=lambda: ContentStream(payload_stream),
        out_content_type=ContentType.TOKENS,
    )

    task: ToTopicModel = ToTopicModel(
        pipeline=pipeline,
        prior=prior,
        corpus_filename=None,
        target_folder="./tests/output",
        target_name=target_name,
        engine=method,
        engine_args=utility.DEFAULT_ENGINE_ARGS,
        store_corpus=True,
        store_compressed=True,
    )

    task.setup()
    task.enter()
    payload: DocumentPayload = next(task.process_stream())

    assert payload is not None
    assert payload.content_type == ContentType.TOPIC_MODEL
    assert isinstance(payload.content, dict)
