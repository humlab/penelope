import penelope.corpus.dtm as dtm
from penelope.corpus import ExtractTaggedTokensOpts, TokensTransformOpts
from penelope.utility import PropertyValueMaskingOpts

from ..pipelines import CorpusPipeline, wildcard


def wildcard_to_DTM_pipeline(
    tokens_transform_opts: TokensTransformOpts = None,
    extract_tagged_tokens_opts: ExtractTaggedTokensOpts = None,
    tagged_tokens_filter_opts: PropertyValueMaskingOpts = None,
    vectorize_opts: dtm.VectorizeOpts = None,
):
    try:
        p: CorpusPipeline = (
            wildcard()
            .tagged_frame_to_tokens(
                extract_opts=extract_tagged_tokens_opts,
                filter_opts=tagged_tokens_filter_opts,
            )
            .tokens_transform(transform_opts=tokens_transform_opts)
            .to_document_content_tuple()
            .tqdm()
            .to_dtm(vectorize_opts=vectorize_opts)
        )
        return p
    except Exception as ex:
        raise ex
