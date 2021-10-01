from __future__ import annotations

from typing import TYPE_CHECKING, Any

from penelope.utility import PropertyValueMaskingOpts, path_add_suffix

from .. import interfaces, pipelines
from ..co_occurrence.pipelines import wildcard_to_partition_by_document_co_occurrence_pipeline
from ..dtm.pipelines import wildcard_to_DTM_pipeline

if TYPE_CHECKING:
    from penelope.co_occurrence import ContextOpts
    from penelope.corpus import TextTransformOpts, TokensTransformOpts, VectorizeOpts
    from penelope.corpus.readers import ExtractTaggedTokensOpts

    # from ..checkpoint.interface import CheckpointOpts
    from ..config import CorpusConfig
    from ..pipelines import CorpusPipeline

# pylint: disable=too-many-locals


def default_done_callback(*_: Any, **__: Any) -> None:
    print("Vectorization done!")


def to_tagged_frame_pipeline(
    *,
    corpus_config: CorpusConfig,
    corpus_filename: str = None,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
    tagged_frames_filename: str = None,
    text_transform_opts: TextTransformOpts = None,
    **_,
) -> CorpusPipeline:
    """Tag corpus using spaCy pipeline. Store result as tagged (pos) data frames

    Args:
        corpus_config (CorpusConfig): [description]
        corpus_filename (str, optional): [description]. Defaults to None.
        enable_checkpoint (bool, optional): [description]. Defaults to True.
        force_checkpoint (bool, optional): [description]. Defaults to False.
        tagged_frames_filename (str, optional): [description]. Defaults to None.

    Raises:
        ex: [description]

    Returns:
        CorpusPipeline: [description]
    """
    try:

        tagged_frame_filename: str = tagged_frames_filename or path_add_suffix(
            corpus_config.pipeline_payload.source, interfaces.DEFAULT_TAGGED_FRAMES_FILENAME_SUFFIX
        )

        pipeline: pipelines.CorpusPipeline = (
            pipelines.CorpusPipeline(config=corpus_config)
            .set_spacy_model(corpus_config.pipeline_payload.memory_store['spacy_model'])
            .load_text(
                reader_opts=corpus_config.text_reader_opts,
                transform_opts=text_transform_opts or corpus_config.text_transform_opts,
                source=corpus_filename,
            )
            .text_to_spacy()
            .spacy_to_pos_tagged_frame()
            .checkpoint(filename=tagged_frame_filename, force_checkpoint=force_checkpoint)
        )

        if enable_checkpoint:
            pipeline = pipeline.checkpoint_feather(
                folder=corpus_config.get_feather_folder(corpus_filename), force=force_checkpoint
            )

        return pipeline

    except Exception as ex:
        raise ex


def spaCy_DTM_pipeline(  # pylint: disable=too-many-arguments
    corpus_config: CorpusConfig,
    extract_opts: ExtractTaggedTokensOpts = None,
    filter_opts: PropertyValueMaskingOpts = None,
    transform_opts: TokensTransformOpts = None,
    vectorize_opts: VectorizeOpts = None,
    corpus_filename: str = None,
    tagged_frames_filename: str = None,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
) -> pipelines.CorpusPipeline:
    try:
        p: pipelines.CorpusPipeline = to_tagged_frame_pipeline(
            corpus_config=corpus_config,
            tagged_frames_filename=tagged_frames_filename,
            corpus_filename=corpus_filename,
            enable_checkpoint=enable_checkpoint,
            force_checkpoint=force_checkpoint,
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
    tagged_frames_filename: str = None,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
) -> pipelines.CorpusPipeline:
    p: pipelines.CorpusPipeline = to_tagged_frame_pipeline(
        corpus_config=corpus_config,
        corpus_filename=corpus_filename,
        tagged_frames_filename=tagged_frames_filename,
        enable_checkpoint=enable_checkpoint,
        force_checkpoint=force_checkpoint,
    ) + wildcard_to_partition_by_document_co_occurrence_pipeline(
        context_opts=context_opts,
        transform_opts=transform_opts,
        extract_opts=extract_opts,
        filter_opts=filter_opts,
        global_tf_threshold=global_threshold_count,
    )
    return p
