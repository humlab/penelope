from penelope.corpus import ExtractTaggedTokensOpts, TokensTransformOpts, VectorizeOpts

from .. import pipelines
from ..config import CorpusConfig


def wildcard_to_DTM_pipeline(
    transform_opts: TokensTransformOpts = None,
    extract_opts: ExtractTaggedTokensOpts = None,
    vectorize_opts: VectorizeOpts = None,
) -> pipelines.CorpusPipeline:
    try:
        p: pipelines.CorpusPipeline = (
            pipelines.wildcard()
            .vocabulary(
                lemmatize=extract_opts.lemmatize,
                progress=True,
                tf_threshold=extract_opts.global_tf_threshold,
                tf_keeps=None,
                close=True,
                to_lower=transform_opts.to_lower,
            )
            .tagged_frame_to_tokens(extract_opts=extract_opts, transform_opts=transform_opts)
            # .tokens_transform(transform_opts=transform_opts)
            .tqdm()
            .to_dtm(vectorize_opts=vectorize_opts)
        )
        return p
    except Exception as ex:
        raise ex


def to_DTM_pipeline(  # pylint: disable=too-many-arguments
    corpus_config: CorpusConfig,
    extract_opts: ExtractTaggedTokensOpts = None,
    transform_opts: TokensTransformOpts = None,
    vectorize_opts: VectorizeOpts = None,
    corpus_source: str = None,
    tagged_corpus_source: str = None,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
) -> pipelines.CorpusPipeline:
    try:
        p: pipelines.CorpusPipeline = pipelines.to_tagged_frame_pipeline(
            corpus_config=corpus_config,
            tagged_corpus_source=tagged_corpus_source,
            corpus_source=corpus_source,
            enable_checkpoint=enable_checkpoint,
            force_checkpoint=force_checkpoint,
        ) + wildcard_to_DTM_pipeline(
            extract_opts=extract_opts,
            transform_opts=transform_opts,
            vectorize_opts=vectorize_opts,
        )
        return p

    except Exception as ex:
        raise ex


def id_tagged_frame_to_DTM_pipeline(
    corpus_config: CorpusConfig,
    corpus_source: str = None,
    id_to_token: bool = False,
    file_pattern: str = None,
    transform_opts: TokensTransformOpts = None,
    extract_opts: ExtractTaggedTokensOpts = None,
    vectorize_opts: VectorizeOpts = None,
) -> pipelines.CorpusPipeline:
    try:
        if corpus_source is None:
            corpus_source = corpus_config.pipeline_payload.source

        file_pattern = file_pattern or corpus_config.get_pipeline_opts_value("tagged_frame_pipeline", "file_pattern")
        if file_pattern is None:
            raise ValueError("file pattern not supplied and not in pipeline opts (corpus config)")

        extract_opts.set_numeric_names()
        vectorize_opts.min_df = extract_opts.global_tf_threshold
        extract_opts.global_tf_threshold = 1
        p: pipelines.CorpusPipeline = (
            pipelines.CorpusPipeline(config=corpus_config)
            .load_id_tagged_frame(
                folder=corpus_source,
                id_to_token=id_to_token,
                file_pattern=file_pattern,
            )
            .filter_tagged_frame(
                extract_opts=extract_opts,
                pos_schema=corpus_config.pos_schema,
                transform_opts=transform_opts,
            )
            .to_dtm(vectorize_opts=vectorize_opts, tagged_column=extract_opts.target_column)
        )
        return p
    except Exception as ex:
        raise ex
