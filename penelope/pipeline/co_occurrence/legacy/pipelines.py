# from penelope.co_occurrence import ContextOpts
# from penelope.corpus import ExtractTaggedTokensOpts, TokensTransformOpts
# from penelope.utility import deprecated

# from ... import pipelines


# @deprecated
# def wildcard_to_partitioned_by_key_co_occurrence_pipeline(
#     *,
#     transform_opts: TokensTransformOpts = None,
#     extract_opts: ExtractTaggedTokensOpts = None,
#     context_opts: ContextOpts = None,
#     tf_threshold: int = None,
# ) -> pipelines.CorpusPipeline:
#     """Computes generic partitioned co-occurrence"""
#     try:
#         pipeline: pipelines.CorpusPipeline = (
#             pipelines.wildcard()
#             .tagged_frame_to_tokens(
#                 extract_opts=extract_opts,
#                 transform_opts=transform_opts,
#             )
#             .vocabulary(lemmatize=extract_opts.lemmatize, progress=True)
#             .tqdm()
#             .to_corpus_co_occurrence(
#                 context_opts=context_opts,
#                 transform_opts=transform_opts,
#             )
#         )

#         return pipeline

#     except Exception as ex:
#         raise ex
