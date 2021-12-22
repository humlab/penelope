from __future__ import annotations

from ... import corpus, utility
from .. import pipelines
from ..config import CorpusConfig


def from_grouped_feather_id_to_tagged_frame_pipeline(
    *,
    corpus_config: CorpusConfig,
    corpus_source: str = None,
    **_,
):
    """Loads a tagged data frame"""

    corpus_source: str = corpus_source or corpus_config.pipeline_payload.source

    p: pipelines.CorpusPipeline = pipelines.CorpusPipeline(config=corpus_config).load_grouped_id_tagged_frame(
        folder=corpus_source,
        to_tagged_frame=True,
        file_pattern='**/prot-*.feather',
    )

    return p


def from_tagged_frame_pipeline(
    *,
    config: CorpusConfig,
    target_name: str,
    corpus_source: str = None,
    train_corpus_folder: str = None,
    target_folder: str = None,
    text_transform_opts: corpus.TextTransformOpts = None,
    extract_opts: corpus.ExtractTaggedTokensOpts = None,
    transform_opts: corpus.TokensTransformOpts = None,
    filter_opts: utility.PropertyValueMaskingOpts = None,
    engine: str = "gensim_lda-multicore",
    engine_args: dict = None,
    store_corpus: bool = False,
    store_compressed: bool = True,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
) -> pipelines.CorpusPipeline:

    corpus_source: str = corpus_source or config.pipeline_payload.source

    p: pipelines.CorpusPipeline = (
        config.get_pipeline(
            "tagged_frame_pipeline",
            corpus_source=corpus_source,
            enable_checkpoint=enable_checkpoint,
            force_checkpoint=force_checkpoint,
            text_transform_opts=text_transform_opts,
        )
        .tagged_frame_to_tokens(
            extract_opts=extract_opts,
            transform_opts=transform_opts,
            filter_opts=filter_opts,
        )
        .to_topic_model(
            corpus_source=None,
            train_corpus_folder=train_corpus_folder,
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
    config: CorpusConfig,
    target_name: str,
    corpus_source: str = None,
    train_corpus_folder: str = None,
    target_folder: str = None,
    text_transform_opts: corpus.TextTransformOpts = None,
    extract_opts: corpus.ExtractTaggedTokensOpts = None,
    transform_opts: corpus.TokensTransformOpts = None,
    filter_opts: utility.PropertyValueMaskingOpts = None,
    engine: str = "gensim_lda-multicore",
    engine_args: dict = None,
    store_corpus: bool = False,
    store_compressed: bool = True,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
) -> pipelines.CorpusPipeline:

    corpus_source: str = corpus_source or config.pipeline_payload.source

    p: pipelines.CorpusPipeline = (
        config.get_pipeline(
            "tagged_frame_pipeline",
            corpus_source=corpus_source,
            enable_checkpoint=enable_checkpoint,
            force_checkpoint=force_checkpoint,
            text_transform_opts=text_transform_opts,
        )
        .to_dtm(
            extract_opts=extract_opts,
            transform_opts=transform_opts,
            filter_opts=filter_opts,
        )
        .to_topic_model(
            corpus_source=None,
            train_corpus_folder=train_corpus_folder,
            target_folder=target_folder,
            target_name=target_name,
            engine=engine,
            engine_args=engine_args,
            store_corpus=store_corpus,
            store_compressed=store_compressed,
        )
    )

    return p
