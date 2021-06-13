from __future__ import annotations

from typing import TYPE_CHECKING

from penelope.co_occurrence import ContextOpts
from penelope.corpus import ExtractTaggedTokensOpts, TokensTransformOpts
from penelope.utility import PropertyValueMaskingOpts

from .. import pipelines

if TYPE_CHECKING:
    from ..pipelines import CorpusPipeline


def wildcard_to_partition_by_document_co_occurrence_pipeline(
    *,
    extract_opts: ExtractTaggedTokensOpts = None,
    filter_opts: PropertyValueMaskingOpts = None,
    transform_opts: TokensTransformOpts = None,
    context_opts: ContextOpts = None,
    global_threshold_count: int = None,
    **kwargs,  # pylint: disable=unused-argument
) -> CorpusPipeline:
    passthroughs: set = context_opts.concept.union(extract_opts.get_passthrough_tokens())
    pipeline: pipelines.CorpusPipeline = (
        pipelines.wildcard()
        .vocabulary(
            lemmatize=extract_opts,
            progress=True,
            tf_threshold=extract_opts.global_tf_threshold,
            tf_keeps=passthroughs,
            close=True,
        )
        .tagged_frame_to_tokens(
            extract_opts=extract_opts.clear_tf_threshold(),
            filter_opts=filter_opts,
            transform_opts=transform_opts,
        )
        # .tokens_transform(transform_opts=transform_opts)
        .to_document_co_occurrence(context_opts=context_opts, ingest_tokens=False)
        .tqdm(desc="Processing documents")
        .to_corpus_co_occurrence(
            context_opts=context_opts,
            global_threshold_count=global_threshold_count,
        )
    )

    return pipeline
