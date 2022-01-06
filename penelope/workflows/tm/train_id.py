from typing import Literal

from penelope import corpus as pc
from penelope import pipeline as pp
from penelope.pipeline.topic_model.pipelines import from_id_tagged_frame_pipeline

# pylint: disable=too-many-arguments


def compute(
    corpus_config: pp.CorpusConfig,
    corpus_source: str,
    target_mode: Literal['train', 'predict', 'both'],
    target_folder: str,
    target_name: str,
    extract_opts: pc.ExtractTaggedTokensOpts,
    vectorize_opts: pc.VectorizeOpts,
    filename_pattern: str = None,
    train_corpus_folder: str = None,
    trained_model_folder: str = None,
    engine: str = "gensim_lda-multicore",
    engine_args: dict = None,
    store_corpus: bool = False,
    store_compressed: bool = True,
    # transform_opts: pc.TokensTransformOpts,
):
    extract_opts.set_numeric_names()
    # tm_pipeline: pp.CorpusPipeline = train_from_id_tagged_frame_pipeline
    # _: dict = config.get_pipeline(
    #     pipeline_key="topic_modeling_pipeline",

    config_opts: dict = corpus_config.get_pipeline_opts('topic_modeling_pipeline')

    filename_pattern = filename_pattern or config_opts.get('file_patter', '**/*.feather')

    value: dict = from_id_tagged_frame_pipeline(
        corpus_config=corpus_config,
        target_mode=target_mode,
        target_folder=target_folder,
        target_name=target_name,
        corpus_source=corpus_source,
        file_pattern=filename_pattern,
        tagged_column=extract_opts.target_column,
        train_corpus_folder=train_corpus_folder,
        trained_model_folder=trained_model_folder,
        extract_opts=extract_opts,
        # transform_opts=transform_opts,
        vectorize_opts=vectorize_opts,
        engine=engine,
        engine_args=engine_args,
        store_corpus=store_corpus,
        store_compressed=store_compressed,
    ).value()

    return value
