from __future__ import annotations

from typing import TYPE_CHECKING

from penelope.co_occurrence import ContextOpts
from penelope.corpus import ExtractTaggedTokensOpts, TokensTransformOpts

from .. import pipelines

if TYPE_CHECKING:
    from ..pipelines import CorpusPipeline


def wildcard_to_partition_by_document_co_occurrence_pipeline(
    *,
    extract_opts: ExtractTaggedTokensOpts = None,
    transform_opts: TokensTransformOpts = None,
    context_opts: ContextOpts = None,
    **kwargs,  # pylint: disable=unused-argument
) -> CorpusPipeline:

    passthroughs: set = context_opts.get_concepts().union(extract_opts.get_passthrough_tokens())
    pipeline: pipelines.CorpusPipeline = (
        pipelines.wildcard()
        .vocabulary(
            lemmatize=extract_opts.lemmatize,
            progress=True,
            tf_threshold=extract_opts.global_tf_threshold,
            tf_keeps=passthroughs,
            close=True,
        )
        .tagged_frame_to_tokens(
            extract_opts=extract_opts,  # .clear_tf_threshold(),
            transform_opts=transform_opts,
        )
        # .tokens_transform(transform_opts=transform_opts)
        .to_document_co_occurrence(context_opts=context_opts)
        # .tqdm(desc="Processing documents")
        .to_corpus_co_occurrence(context_opts=context_opts)
    )

    return pipeline
