from penelope.co_occurrence import ContextOpts
from penelope.corpus import ExtractTaggedTokensOpts, TokensTransformOpts
from penelope.utility import PropertyValueMaskingOpts, deprecated

from ... import pipelines


@deprecated
def wildcard_to_partitioned_by_key_co_occurrence_pipeline(
    *,
    transform_opts: TokensTransformOpts = None,
    extract_opts: ExtractTaggedTokensOpts = None,
    filter_opts: PropertyValueMaskingOpts = None,
    context_opts: ContextOpts = None,
    global_threshold_count: int = None,
) -> pipelines.CorpusPipeline:
    """Computes generic partitioned co-occurrence"""
    try:
        pipeline: pipelines.CorpusPipeline = (
            pipelines.wildcard()
            .tagged_frame_to_tokens(
                extract_opts=extract_opts,
                filter_opts=filter_opts,
                transform_opts=transform_opts,
            )
            # .tokens_transform(
            #     transform_opts=TokensTransformOpts(
            #         to_lower=transform_opts.to_lower,
            #     ),
            # )
            .vocabulary()
            .to_document_content_tuple()
            .tqdm()
            # .tap_stream("./tests/output/tapped_stream__prior_to_co_occurrence.zip",  "tap_4_prior_to_co_occurrence")
            .to_corpus_co_occurrence(
                context_opts=context_opts,
                transform_opts=transform_opts,
                global_threshold_count=global_threshold_count,
            )
        )

        return pipeline

    except Exception as ex:
        raise ex
