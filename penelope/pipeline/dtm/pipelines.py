from penelope.corpus import ExtractTaggedTokensOpts, TokensTransformOpts, VectorizeOpts
from penelope.utility import PropertyValueMaskingOpts

from ..pipelines import CorpusPipeline, wildcard


def wildcard_to_DTM_pipeline(
    transform_opts: TokensTransformOpts = None,
    extract_opts: ExtractTaggedTokensOpts = None,
    filter_opts: PropertyValueMaskingOpts = None,
    vectorize_opts: VectorizeOpts = None,
):
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
            .tagged_frame_to_tokens(extract_opts=extract_opts, filter_opts=filter_opts, transform_opts=transform_opts)
            # .tokens_transform(transform_opts=transform_opts)
            .to_document_content_tuple()
            .tqdm()
            .to_dtm(vectorize_opts=vectorize_opts)
        )
        return p
    except Exception as ex:
        raise ex
