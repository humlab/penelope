from __future__ import annotations

from penelope.corpus.dtm.vectorizer import VectorizeOpts

from ... import corpus
from .. import pipelines
from ..config import CorpusConfig


def load_id_tagged_frame_pipeline(
    *,
    corpus_config: CorpusConfig,
    corpus_source: str = None,
    id_to_token: bool = False,
    file_pattern: str = '**/prot-*.feather',
    **_,
):
    """Loads a tagged data frame"""

    corpus_source: str = corpus_source or corpus_config.pipeline_payload.source

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
    target_name: str,
    train_corpus_folder: str = None,
    target_folder: str = None,
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
    corpus_config: CorpusConfig,
    corpus_source: str = None,
    extract_opts: corpus.ExtractTaggedTokensOpts = None,
    transform_opts: corpus.TokensTransformOpts = None,
    file_pattern: str = '**/prot-*.feather',
    target_name: str,
    train_corpus_folder: str = None,
    target_folder: str = None,
    engine: str = "gensim_lda-multicore",
    engine_args: dict = None,
    store_corpus: bool = False,
    store_compressed: bool = True,
    **_,
) -> pipelines.CorpusPipeline:

    corpus_source: str = corpus_source or corpus_config.pipeline_payload.source
    vectorize_opts: VectorizeOpts = VectorizeOpts(
        already_tokenized=True,
        lowercase=False,
    )
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
