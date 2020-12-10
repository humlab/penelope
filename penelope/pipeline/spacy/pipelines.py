from __future__ import annotations

from typing import TYPE_CHECKING

from penelope.co_occurrence.co_occurrence import ContextOpts
from penelope.utility import get_logger

from ..pipelines import CorpusPipeline

if TYPE_CHECKING:
    from penelope.corpus import TokensTransformOpts, VectorizeOpts
    from penelope.corpus.readers import ExtractTaggedTokensOpts, TaggedTokensFilterOpts, TextTransformOpts

    from ..config import CorpusConfig


logger = get_logger()
# pylint: disable=too-many-locals


class SpacyPipeline(CorpusPipeline):
    pass


def default_done_callback(*_, **__):
    print("Vectorization done!")


def spaCy_to_pos_tagged_frame_pipeline(
    corpus_config: CorpusConfig,
    checkpoint_filename: str = None,
):
    try:

        # checkpoint_filename: str = path_add_suffix(corpus_filename, '_pos_csv')

        pipeline: SpacyPipeline = (
            SpacyPipeline(payload=corpus_config.pipeline_payload)
            .set_spacy_model(corpus_config.pipeline_payload.memory_store['spacy_model'])
            .load_text(reader_opts=corpus_config.text_reader_opts, transform_opts=TextTransformOpts())
            .text_to_spacy()
            .tqdm()
            .passthrough()
            .spacy_to_pos_tagged_frame()
            .checkpoint(checkpoint_filename)
        )
        return pipeline

    except Exception as ex:
        raise ex


def spaCy_DTM_pipeline(
    corpus_config: CorpusConfig,
    tokens_transform_opts: TokensTransformOpts = None,
    extract_tagged_tokens_opts: ExtractTaggedTokensOpts = None,
    tagged_tokens_filter_opts: TaggedTokensFilterOpts = None,
    vectorize_opts: VectorizeOpts = None,
    checkpoint_filename: str = None,
):
    try:

        # checkpoint_filename: str = path_add_suffix(corpus_filename, '_pos_csv')
        pipeline: SpacyPipeline = (
            spaCy_to_pos_tagged_frame_pipeline(
                corpus_config=corpus_config,
                checkpoint_filename=checkpoint_filename,
            )
            .tagged_frame_to_tokens(extract_opts=extract_tagged_tokens_opts, filter_opts=tagged_tokens_filter_opts)
            .tokens_transform(tokens_transform_opts=tokens_transform_opts)
            # .tokens_to_text()
            .to_document_content_tuple()
            .tqdm()
            .to_dtm(vectorize_opts=vectorize_opts)
        )
        return pipeline

    except Exception as ex:
        raise ex


def spaCy_co_occurrence_pipeline(
    corpus_config: CorpusConfig,
    tokens_transform_opts: TokensTransformOpts = None,
    extract_tagged_tokens_opts: ExtractTaggedTokensOpts = None,
    tagged_tokens_filter_opts: TaggedTokensFilterOpts = None,
    context_opts: ContextOpts = None,
    global_threshold_count: int = None,
    partition_column: str = 'year',
    checkpoint_filename: str = None,
):
    try:
        pipeline: SpacyPipeline = (
            spaCy_to_pos_tagged_frame_pipeline(
                corpus_config=corpus_config,
                checkpoint_filename=checkpoint_filename,
            )
            .tagged_frame_to_tokens(extract_opts=extract_tagged_tokens_opts, filter_opts=tagged_tokens_filter_opts)
            .tokens_transform(tokens_transform_opts=tokens_transform_opts)
            .vocabulary()
            .to_document_content_tuple()
            .to_co_occurrence(
                context_opts=context_opts,
                global_threshold_count=global_threshold_count,
                partition_column=partition_column,
            )
            .tqdm()
        )
        return pipeline

    except Exception as ex:
        raise ex
