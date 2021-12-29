from penelope import corpus as pc
from penelope import pipeline as pp
from penelope.pipeline.topic_model.pipelines import from_id_tagged_frame_pipeline

# pylint: disable=too-many-arguments


def compute(
    corpus_config: pp.CorpusConfig,
    corpus_source: str,
    # transform_opts: pc.TokensTransformOpts,
    extract_opts: pc.ExtractTaggedTokensOpts,
    target_folder: str,
    target_name: str,
    engine: str = "gensim_lda-multicore",
    engine_args: dict = None,
    store_corpus: bool = False,
    store_compressed: bool = True,
    train_corpus_folder: str = None,
):
    extract_opts.set_numeric_names()

    # tm_pipeline: pp.CorpusPipeline = from_id_tagged_frame_pipeline
    # _: dict = config.get_pipeline(
    #     pipeline_key="topic_modeling_pipeline",

    _: dict = from_id_tagged_frame_pipeline(
        corpus_config=corpus_config,
        target_folder=target_folder,
        target_name=target_name,
        corpus_source=corpus_source,
        tagged_column=extract_opts.target_column,
        train_corpus_folder=train_corpus_folder,
        extract_opts=extract_opts,
        # transform_opts=transform_opts,
        engine=engine,
        engine_args=engine_args,
        store_corpus=store_corpus,
        store_compressed=store_compressed,
    ).value()
