from __future__ import annotations

from ctypes import ArgumentError
from typing import Literal

from penelope.corpus.dtm.vectorizer import VectorizeOpts

from ... import corpus
from .. import pipelines
from ..config import CorpusConfig


def load_id_tagged_frame_pipeline(
    *,
    corpus_config: CorpusConfig,
    corpus_source: str = None,
    id_to_token: bool = False,
    file_pattern: str = None,
    **_,
):
    """Loads a tagged data frame"""

    corpus_source: str = corpus_source or corpus_config.pipeline_payload.source

    file_pattern = file_pattern or corpus_config.get_pipeline_opts_value("tagged_frame_pipeline", "file_pattern")

    if file_pattern is None:
        raise ArgumentError("file pattern not supplied and not in pipeline opts (corpus config)")

    p: pipelines.CorpusPipeline = pipelines.CorpusPipeline(config=corpus_config).load_id_tagged_frame(
        folder=corpus_source,
        id_to_token=id_to_token,
        file_pattern=file_pattern,
    )

    return p


def from_tagged_frame_pipeline(
    *,
    corpus_config: CorpusConfig,
    corpus_source: str = None,
    train_corpus_folder: str = None,
    trained_model_folder: str = None,
    target_mode: Literal['train', 'predict', 'both'] = 'both',
    target_folder: str = None,
    target_name: str,
    text_transform_opts: corpus.TextTransformOpts = None,
    extract_opts: corpus.ExtractTaggedTokensOpts = None,
    transform_opts: corpus.TokensTransformOpts = None,
    engine: str = "gensim_lda-multicore",
    engine_args: dict = None,
    store_corpus: bool = False,
    store_compressed: bool = True,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
    **_,
) -> pipelines.CorpusPipeline:

    corpus_source: str = corpus_source or corpus_config.pipeline_payload.source

    p: pipelines.CorpusPipeline = (
        corpus_config.get_pipeline(
            "tagged_frame_pipeline",
            corpus_source=corpus_source,
            enable_checkpoint=enable_checkpoint,
            force_checkpoint=force_checkpoint,
            text_transform_opts=text_transform_opts,
        )
        .tagged_frame_to_tokens(
            extract_opts=extract_opts,
            transform_opts=transform_opts,
        )
        .to_topic_model(
            train_corpus_folder=train_corpus_folder,
            trained_model_folder=trained_model_folder,
            target_mode=target_mode,
            target_folder=target_folder,
            target_name=target_name,
            engine=engine,
            engine_args=engine_args,
            store_corpus=store_corpus,
            store_compressed=store_compressed,
        )
    )

    return p


def from_id_tagged_frame_pipeline(
    *,
    corpus_config: CorpusConfig,
    corpus_source: str = None,
    train_corpus_folder: str = None,
    trained_model_folder: str = None,
    target_mode: Literal['train', 'predict', 'both'] = 'both',
    target_folder: str = None,
    target_name: str,
    extract_opts: corpus.ExtractTaggedTokensOpts = None,
    transform_opts: corpus.TokensTransformOpts = None,
    vectorize_opts: VectorizeOpts = None,
    file_pattern: str = '**/*.feather',
    engine: str = "gensim_lda-multicore",
    engine_args: dict = None,
    store_corpus: bool = False,
    store_compressed: bool = True,
    **_,
) -> pipelines.CorpusPipeline:

    corpus_source: str = corpus_source or corpus_config.pipeline_payload.source
    vectorize_opts: VectorizeOpts = vectorize_opts or VectorizeOpts()
    vectorize_opts.update(already_tokenized=True, lowercase=False, min_tf=extract_opts.global_tf_threshold)
    extract_opts.global_tf_threshold = 1
    p: pipelines.CorpusPipeline = (
        load_id_tagged_frame_pipeline(
            corpus_config=corpus_config,
            corpus_source=corpus_source,
            id_to_token=False,
            file_pattern=file_pattern,
        )
        .filter_tagged_frame(
            extract_opts=extract_opts,
            transform_opts=transform_opts,
            pos_schema=corpus_config.pos_schema,
        )
        .to_dtm(
            vectorize_opts=vectorize_opts,
            tagged_column=extract_opts.target_column,
        )
        .to_topic_model(
            train_corpus_folder=train_corpus_folder,
            trained_model_folder=trained_model_folder,
            target_mode=target_mode,
            target_folder=target_folder,
            target_name=target_name,
            engine=engine,
            engine_args=engine_args,
            store_corpus=store_corpus,
            store_compressed=store_compressed,
        )
    )

    return p
