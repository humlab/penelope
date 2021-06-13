from __future__ import annotations

from typing import TYPE_CHECKING, Any

from penelope.utility import PropertyValueMaskingOpts, path_add_suffix

from .. import pipelines
from ..co_occurrence.pipelines import wildcard_to_partition_by_document_co_occurrence_pipeline
from ..dtm.pipelines import wildcard_to_DTM_pipeline

if TYPE_CHECKING:
    from penelope.co_occurrence import ContextOpts
    from penelope.corpus import TokensTransformOpts, VectorizeOpts
    from penelope.corpus.readers import ExtractTaggedTokensOpts

    from ..config import CorpusConfig
    from ..pipelines import CorpusPipeline
# pylint: disable=too-many-locals


def default_done_callback(*_: Any, **__: Any) -> None:
    print("Vectorization done!")


def to_tagged_frame_pipeline(
    corpus_config: CorpusConfig,
    corpus_filename: str = None,
    checkpoint_filename: str = None,
) -> CorpusPipeline:
    try:

        _checkpoint_filename: str = checkpoint_filename or path_add_suffix(
            corpus_config.pipeline_payload.source, '_pos_csv'
        )

        pipeline: pipelines.CorpusPipeline = (
            pipelines.CorpusPipeline(config=corpus_config)
            .set_spacy_model(corpus_config.pipeline_payload.memory_store['spacy_model'])
            .load_text(
                reader_opts=corpus_config.text_reader_opts,
                transform_opts=None,
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
    extract_opts: ExtractTaggedTokensOpts = None,
    filter_opts: PropertyValueMaskingOpts = None,
    transform_opts: TokensTransformOpts = None,
    vectorize_opts: VectorizeOpts = None,
    corpus_filename: str = None,
    checkpoint_filename: str = None,
) -> pipelines.CorpusPipeline:
    try:
        p: pipelines.CorpusPipeline = to_tagged_frame_pipeline(
            corpus_config=corpus_config,
            checkpoint_filename=checkpoint_filename,
            corpus_filename=corpus_filename,
        ) + wildcard_to_DTM_pipeline(
            extract_opts=extract_opts,
            filter_opts=filter_opts,
            transform_opts=transform_opts,
            vectorize_opts=vectorize_opts,
        )
        return p

    except Exception as ex:
        raise ex


# pylint: disable=too-many-arguments
def spaCy_co_occurrence_pipeline(
    corpus_config: CorpusConfig,
    corpus_filename: str,
    transform_opts: TokensTransformOpts = None,
    extract_opts: ExtractTaggedTokensOpts = None,
    filter_opts: PropertyValueMaskingOpts = None,
    context_opts: ContextOpts = None,
    global_threshold_count: int = None,
    checkpoint_filename: str = None,
) -> pipelines.CorpusPipeline:
    p: pipelines.CorpusPipeline = to_tagged_frame_pipeline(
        corpus_config=corpus_config,
        corpus_filename=corpus_filename,
        checkpoint_filename=checkpoint_filename,
    ) + wildcard_to_partition_by_document_co_occurrence_pipeline(
        context_opts=context_opts,
        transform_opts=transform_opts,
        extract_opts=extract_opts,
        filter_opts=filter_opts,
        global_threshold_count=global_threshold_count,
    )
    return p
