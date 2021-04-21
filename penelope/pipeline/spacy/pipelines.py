from __future__ import annotations

from typing import TYPE_CHECKING, Any

from penelope.co_occurrence.co_occurrence import ContextOpts
from penelope.corpus.readers.text_transformer import TextTransformOpts
from penelope.utility import PropertyValueMaskingOpts, get_logger, path_add_suffix

from .. import pipelines

if TYPE_CHECKING:
    from penelope.corpus import TokensTransformOpts, VectorizeOpts
    from penelope.corpus.readers import ExtractTaggedTokensOpts

    from ..config import CorpusConfig


logger = get_logger()
# pylint: disable=too-many-locals


def default_done_callback(*_: Any, **__: Any) -> None:
    print("Vectorization done!")


def to_tagged_frame_pipeline(
    corpus_config: CorpusConfig,
    corpus_filename: str = None,
    checkpoint_filename: str = None,
) -> pipelines.CorpusPipeline:
    try:

        _checkpoint_filename: str = checkpoint_filename or path_add_suffix(
            corpus_config.pipeline_payload.source, '_pos_csv'
        )

        pipeline: pipelines.CorpusPipeline = (
            pipelines.CorpusPipeline(config=corpus_config)
            .set_spacy_model(corpus_config.pipeline_payload.memory_store['spacy_model'])
            .load_text(
                reader_opts=corpus_config.text_reader_opts,
                transform_opts=TextTransformOpts(),
                source=corpus_filename,
            )
            .text_to_spacy()
            .tqdm()
            .passthrough()
            .spacy_to_pos_tagged_frame()
            .checkpoint(_checkpoint_filename)
        )
        return pipeline

    except Exception as ex:
        raise ex


def spaCy_DTM_pipeline(
    corpus_config: CorpusConfig,
    extract_tagged_tokens_opts: ExtractTaggedTokensOpts = None,
    tagged_tokens_filter_opts: PropertyValueMaskingOpts = None,
    tokens_transform_opts: TokensTransformOpts = None,
    vectorize_opts: VectorizeOpts = None,
    corpus_filename: str = None,
    checkpoint_filename: str = None,
) -> pipelines.CorpusPipeline:
    try:
        p: pipelines.CorpusPipeline = to_tagged_frame_pipeline(
            corpus_config=corpus_config, checkpoint_filename=checkpoint_filename, corpus_filename=corpus_filename
        ) + pipelines.wildcard_to_DTM_pipeline(
            extract_tagged_tokens_opts=extract_tagged_tokens_opts,
            tagged_tokens_filter_opts=tagged_tokens_filter_opts,
            tokens_transform_opts=tokens_transform_opts,
            vectorize_opts=vectorize_opts,
        )
        return p

    except Exception as ex:
        raise ex


# pylint: disable=too-many-arguments
def spaCy_co_occurrence_pipeline(
    corpus_config: CorpusConfig,
    corpus_filename: str,
    tokens_transform_opts: TokensTransformOpts = None,
    extract_tagged_tokens_opts: ExtractTaggedTokensOpts = None,
    tagged_tokens_filter_opts: PropertyValueMaskingOpts = None,
    context_opts: ContextOpts = None,
    global_threshold_count: int = None,
    partition_column: str = 'year',
    checkpoint_filename: str = None,
) -> pipelines.CorpusPipeline:
    try:
        p: pipelines.CorpusPipeline = to_tagged_frame_pipeline(
            corpus_config=corpus_config,
            corpus_filename=corpus_filename,
            checkpoint_filename=checkpoint_filename,
        ) + pipelines.wildcard_to_co_occurrence_pipeline(
            context_opts=context_opts,
            tokens_transform_opts=tokens_transform_opts,
            extract_tagged_tokens_opts=extract_tagged_tokens_opts,
            tagged_tokens_filter_opts=tagged_tokens_filter_opts,
            global_threshold_count=global_threshold_count,
            partition_column=partition_column,
        )
        return p

    except Exception as ex:
        raise ex
