import os

import penelope.corpus.dtm as dtm
from penelope.notebook.interface import ComputeOpts
from penelope.pipeline.config import CorpusConfig
from penelope.pipeline.pipelines import wildcard_to_DTM_pipeline

CheckpointPath = str


def compute(
    args: ComputeOpts,
    corpus_config: CorpusConfig,
) -> dtm.VectorizedCorpus:

    try:

        assert args.is_satisfied()

        corpus: dtm.VectorizedCorpus = (
            corpus_config.get_pipeline("tagged_frame_pipeline")
            + wildcard_to_DTM_pipeline(
                tokens_transform_opts=args.tokens_transform_opts,
                extract_tagged_tokens_opts=args.extract_tagged_tokens_opts,
                tagged_tokens_filter_opts=args.tagged_tokens_filter_opts,
                vectorize_opts=args.vectorize_opts,
            )
        ).value()

        if (args.count_threshold or 1) > 1:
            corpus = corpus.slice_by_n_count(args.count_threshold)

        if args.persist:
            store_corpus_bundle(corpus, args)

        return corpus

    except Exception as ex:
        raise ex


def store_corpus_bundle(corpus: dtm.VectorizedCorpus, args: ComputeOpts):

    if dtm.VectorizedCorpus.dump_exists(tag=args.corpus_tag, folder=args.target_folder):
        dtm.VectorizedCorpus.remove(tag=args.corpus_tag, folder=args.target_folder)

    target_folder = args.target_folder

    if args.create_subfolder:
        if os.path.split(target_folder)[1] != args.corpus_tag:
            target_folder = os.path.join(target_folder, args.corpus_tag)
        os.makedirs(target_folder, exist_ok=True)

    corpus.dump(tag=args.corpus_tag, folder=target_folder)

    dtm.VectorizedCorpus.dump_options(
        tag=args.corpus_tag,
        folder=target_folder,
        options={
            'input_filename': args.corpus_filename,
            'output_folder': target_folder,
            'output_tag': args.corpus_tag,
            'count_threshold': args.count_threshold,
            'reader_opts': args.text_reader_opts.props,
            'tokens_transform_opts': args.tokens_transform_opts.props,
            'extract_tokens_opts': args.extract_tagged_tokens_opts.props
            if args.extract_tagged_tokens_opts is not None
            else {},
            'vectorize_opt': args.vectorize_opts.props,
        },
    )
