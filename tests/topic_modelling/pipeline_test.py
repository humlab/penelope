import os
import uuid
from unittest.mock import MagicMock, Mock

import pytest

from penelope.corpus import ExtractTaggedTokensOpts, TokensTransformOpts
from penelope.corpus.dtm.vectorizer import VectorizeOpts
from penelope.pipeline import ContentType, CorpusConfig, CorpusPipeline, DocumentPayload, ITask
from penelope.pipeline.interfaces import ContentStream
from penelope.pipeline.topic_model.tasks import ToTopicModel
from penelope.topic_modelling.utility import find_models
from penelope.vendor import gensim_api
from tests.fixtures import TranströmerCorpus
from tests.pipeline.fixtures import SPARV_TAGGED_COLUMNS

# pylint: disable=redefined-outer-name


def tranströmer_topic_model_payload(method: str, target_folder: str, target_name: str) -> DocumentPayload:
    transform_opts: TokensTransformOpts = TokensTransformOpts()
    extract_opts: ExtractTaggedTokensOpts = ExtractTaggedTokensOpts(
        lemmatize=True,
        pos_includes='',
        pos_excludes='MAD|MID|PAD',
        text_column='token',
        lemma_column='baseform',
        pos_column='pos',
    )
    default_engine_args: dict = {
        'n_topics': 4,
        'passes': 1,
        'random_seed': 42,
        'workers': 1,
        'max_iter': 100,
        'work_folder': os.path.join(target_folder, target_name),
    }
    config: CorpusConfig = CorpusConfig.load('./tests/test_data/tranströmer.yml')
    corpus_source: str = './tests/test_data/tranströmer_corpus_pos_csv.zip'
    p: CorpusPipeline = (
        CorpusPipeline(config=config)
        .load_tagged_frame(
            filename=corpus_source,
            checkpoint_opts=config.checkpoint_opts,
            extra_reader_opts=config.text_reader_opts,
        )
        .tagged_frame_to_tokens(extract_opts=extract_opts, transform_opts=transform_opts)
        .to_dtm(VectorizeOpts(already_tokenized=True))
        .to_topic_model(
            target_mode='both',
            target_folder=target_folder,
            target_name=target_name,
            engine=method,
            engine_args=default_engine_args,
            store_corpus=True,
            store_compressed=True,
        )
    )

    payload: DocumentPayload = p.single()

    return payload


@pytest.mark.skipif(not gensim_api.GENSIM_INSTALLED, reason="Gensim not installed")
@pytest.mark.long_running
@pytest.mark.parametrize('method', ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_predict_topics(method: str):

    """Train a model that will be used in prediction"""

    target_folder: str = './tests/output'
    train_target_name: str = f'train_{str(uuid.uuid1())[:8]}'
    payload: DocumentPayload = tranströmer_topic_model_payload(
        method=method, target_folder=target_folder, target_name=train_target_name
    )
    model_folder: str = os.path.join(payload.content.get("target_folder"), payload.content.get("target_name"))

    """Predict using trained model"""

    config: CorpusConfig = CorpusConfig.load('./tests/test_data/tranströmer.yml')
    corpus_source: str = './tests/test_data/tranströmer_corpus_pos_csv.zip'
    minimum_probability: float = 0.001
    n_tokens: int = 100
    predict_target_name: str = f'predict_{str(uuid.uuid1())[:8]}'
    transform_opts = TokensTransformOpts()
    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True,
        pos_includes='',
        pos_excludes='MAD|MID|PAD',
        **config.checkpoint_opts.tagged_columns,
    )
    vectorize_opts: VectorizeOpts = VectorizeOpts(already_tokenized=True)
    payload: DocumentPayload = (
        CorpusPipeline(config=config)
        .load_tagged_frame(
            filename=corpus_source,
            checkpoint_opts=config.checkpoint_opts,
            extra_reader_opts=config.text_reader_opts,
        )
        .tagged_frame_to_tokens(extract_opts=extract_opts, transform_opts=transform_opts)
        .to_dtm(vectorize_opts=vectorize_opts)
        .predict_topics(
            model_folder=model_folder,
            target_folder=target_folder,
            target_name=predict_target_name,
            minimum_probability=minimum_probability,
            n_tokens=n_tokens,
        )
    ).single()

    assert payload is not None

    model_infos = find_models('./tests/output')
    assert any(m['name'] == predict_target_name for m in model_infos)
    model_info = next(m for m in model_infos if m['name'] == predict_target_name)
    assert 'method' in model_info['options']


@pytest.mark.skipif(not gensim_api.GENSIM_INSTALLED, reason="Gensim not installed")
@pytest.mark.long_running
@pytest.mark.parametrize("method", ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_topic_model_task_with_token_stream_and_document_index(method):

    target_folder: str = './tests/output'
    target_name: str = f'{str(uuid.uuid1())[:8]}'
    corpus = TranströmerCorpus()
    default_engine_args: dict = {
        'n_topics': 4,
        'passes': 1,
        'random_seed': 42,
        'workers': 1,
        'max_iter': 100,
        'work_folder': os.path.join(target_folder, target_name),
    }
    payload_stream = lambda: [
        DocumentPayload(content_type=ContentType.TOKENS, filename=filename, content=tokens)
        for filename, tokens in corpus
    ]

    pipeline = Mock(
        spec=CorpusPipeline,
        **{
            'payload.memory_store': SPARV_TAGGED_COLUMNS,
            'payload.document_index': corpus.document_index,
            'payload.token2id': None,
        },
    )

    prior = MagicMock(
        spec=ITask,
        outstream=payload_stream,
        content_stream=lambda: ContentStream(payload_stream),
        out_content_type=ContentType.TOKENS,
        filename_content_stream=lambda: [(p.filename, p.content) for p in payload_stream()],
    )

    task: ToTopicModel = ToTopicModel(
        pipeline=pipeline,
        prior=prior,
        target_folder="./tests/output",
        target_name=target_name,
        engine=method,
        engine_args=default_engine_args,
        store_corpus=True,
        store_compressed=True,
    )
    task.resolved_prior_out_content_type = lambda: ContentType.TOKENS

    task.setup()
    task.enter()
    payload: DocumentPayload = next(task.process_stream())

    assert payload is not None
    assert payload.content_type == ContentType.TOPIC_MODEL
    assert isinstance(payload.content, dict)

    output_models = find_models('./tests/output')
    assert any(m['name'] == target_name for m in output_models)
