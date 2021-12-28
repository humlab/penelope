from penelope.corpus import ExtractTaggedTokensOpts, TokensTransformOpts, VectorizeOpts

from ..pipelines import CorpusPipeline, wildcard


def wildcard_to_DTM_pipeline(
    transform_opts: TokensTransformOpts = None,
    extract_opts: ExtractTaggedTokensOpts = None,
    vectorize_opts: VectorizeOpts = None,
) -> CorpusPipeline:
    try:
        p: CorpusPipeline = (
            wildcard()
            .vocabulary(
                lemmatize=extract_opts,
                progress=True,
                tf_threshold=extract_opts.global_tf_threshold,
                tf_keeps=None,
                close=True,
            )
            .tagged_frame_to_tokens(extract_opts=extract_opts, transform_opts=transform_opts)
            # .tokens_transform(transform_opts=transform_opts)
            .tqdm()
            .to_dtm(vectorize_opts=vectorize_opts)
        )
        return p
    except Exception as ex:
        raise ex
