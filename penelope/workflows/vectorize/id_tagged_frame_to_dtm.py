from typing import Any

from penelope import pipeline
from penelope.corpus import VectorizedCorpus
from penelope.notebook.interface import ComputeOpts
from penelope.pipeline import CorpusConfig, id_tagged_frame_to_DTM_pipeline

from .dtm import store_corpus_bundle

CheckpointPath = str


def compute(
    args: ComputeOpts,
    corpus_config: CorpusConfig,
) -> VectorizedCorpus:

    try:

        assert args.is_satisfied()

        corpus: VectorizedCorpus = id_tagged_frame_to_DTM_pipeline(
            corpus_config=corpus_config,
            corpus_source=args.corpus_source,
            extract_opts=args.extract_opts,
            file_pattern=args.text_reader_opts.filename_filter,
            id_to_token=False,
            transform_opts=args.transform_opts,
            vectorize_opts=args.vectorize_opts,
        ).value()

        if (args.tf_threshold or 1) > 1:
            corpus = corpus.slice_by_tf(args.tf_threshold)

        if args.persist:
            store_corpus_bundle(corpus, args)

        return corpus

    except Exception as ex:
        raise ex
