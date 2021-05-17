from penelope.co_occurrence import ContextOpts
from penelope.corpus import ExtractTaggedTokensOpts, TokensTransformOpts
from penelope.utility import PropertyValueMaskingOpts, deprecated

from .. import pipelines


def wildcard_to_partition_by_document_co_occurrence_pipeline(
    *,
    extract_tagged_tokens_opts: ExtractTaggedTokensOpts = None,
    tagged_tokens_filter_opts: PropertyValueMaskingOpts = None,
    tokens_transform_opts: TokensTransformOpts = None,  # pylint: disable=unused-argument
    context_opts: ContextOpts = None,
    global_threshold_count: int = None,
    ignore_pad: bool = False,
    **kwargs,  # pylint: disable=unused-argument
):
    try:
        # FIXME: tokens_transform_opts ignored
        pipeline: pipelines.CorpusPipeline = (
            pipelines.wildcard()
            .tagged_frame_to_tokens(
                extract_opts=extract_tagged_tokens_opts,
                filter_opts=tagged_tokens_filter_opts,
            )
            # .tokens_transform(tokens_transform_opts=transform_opts)
            .to_document_co_occurrence(context_opts=context_opts, ingest_tokens=True)
            .tqdm()
            .to_corpus_document_co_occurrence(
                context_opts=context_opts,
                global_threshold_count=global_threshold_count,
                ignore_pad=ignore_pad,
            )
        )

        return pipeline

    except Exception as ex:
        raise ex


@deprecated
def wildcard_to_partitioned_by_key_co_occurrence_pipeline(
    *,
    tokens_transform_opts: TokensTransformOpts = None,
    extract_tagged_tokens_opts: ExtractTaggedTokensOpts = None,
    tagged_tokens_filter_opts: PropertyValueMaskingOpts = None,
    context_opts: ContextOpts = None,
    global_threshold_count: int = None,
):
    """Computes generic partitioned co-occurrence"""
    try:
        pipeline: pipelines.CorpusPipeline = (
            pipelines.wildcard()
            .tagged_frame_to_tokens(
                extract_opts=extract_tagged_tokens_opts,
                filter_opts=tagged_tokens_filter_opts,
            )
            # .tap_stream("./tests/output/tapped_stream__tagged_frame_to_tokens.zip",  "tap_2_tagged_frame_to_tokens")
            .tokens_transform(
                tokens_transform_opts=TokensTransformOpts(
                    to_lower=tokens_transform_opts.to_lower,
                ),
            )
            # .tap_stream("./tests/output/tapped_stream__tokens_transform.zip",  "tap_3_tokens_transform")
            .vocabulary()
            .to_document_content_tuple()
            .tqdm()
            # .tap_stream("./tests/output/tapped_stream__prior_to_co_occurrence.zip",  "tap_4_prior_to_co_occurrence")
            .to_corpus_co_occurrence(
                context_opts=context_opts,
                transform_opts=tokens_transform_opts,
                global_threshold_count=global_threshold_count,
            )
        )

        return pipeline

    except Exception as ex:
        raise ex
