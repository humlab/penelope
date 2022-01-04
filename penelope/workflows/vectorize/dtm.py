import os

from penelope import pipeline
from penelope.corpus import VectorizedCorpus
from penelope.pipeline.config import CorpusConfig
from penelope.pipeline.dtm.pipelines import wildcard_to_DTM_pipeline
from penelope.workflows import interface

CheckpointPath = str


def compute(
    args: interface.ComputeOpts,
    corpus_config: CorpusConfig,
    tagged_frame_pipeline: pipeline.CorpusPipeline = None,
) -> VectorizedCorpus:

    try:

        assert args.is_satisfied()

        if tagged_frame_pipeline is None:
            tagged_frame_pipeline = corpus_config.get_pipeline(
                "tagged_frame_pipeline",
                corpus_source=args.corpus_source,
                enable_checkpoint=args.enable_checkpoint,
                force_checkpoint=args.force_checkpoint,
                tagged_corpus_source=args.tagged_corpus_source,
            )
        corpus: VectorizedCorpus = (
            tagged_frame_pipeline
            + wildcard_to_DTM_pipeline(
                transform_opts=args.transform_opts,
                extract_opts=args.extract_opts,
                vectorize_opts=args.vectorize_opts,
            )
        ).value()

        if (args.tf_threshold or 1) > 1:
            corpus = corpus.slice_by_tf(args.tf_threshold)

        if args.persist:
            store_corpus_bundle(corpus, args)

        return corpus

    except Exception as ex:
        raise ex


def store_corpus_bundle(corpus: VectorizedCorpus, args: interface.ComputeOpts):

    if VectorizedCorpus.dump_exists(tag=args.corpus_tag, folder=args.target_folder):
        VectorizedCorpus.remove(tag=args.corpus_tag, folder=args.target_folder)

    target_folder = args.target_folder

    if args.create_subfolder:
        if os.path.split(target_folder)[1] != args.corpus_tag:
            target_folder = os.path.join(target_folder, args.corpus_tag)
        os.makedirs(target_folder, exist_ok=True)

    corpus.dump(tag=args.corpus_tag, folder=target_folder)

    VectorizedCorpus.dump_options(
        tag=args.corpus_tag,
        folder=target_folder,
        options=args.props,
    )
