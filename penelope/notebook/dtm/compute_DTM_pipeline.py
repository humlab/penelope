from typing import Callable

import penelope.corpus.dtm as dtm
import penelope.pipeline as pipeline
from penelope.utility import path_add_suffix

from ..interface import ComputeOpts
from .store import store_corpus_bundle


def compute_document_term_matrix(
    corpus_config: pipeline.CorpusConfig,
    pipeline_factory: Callable,
    args: ComputeOpts,
) -> dtm.VectorizedCorpus:

    try:

        assert args.is_satisfied()

        checkpoint_filename: str = path_add_suffix(args.corpus_filename, '_pos_csv')
        corpus: dtm.VectorizedCorpus = pipeline_factory(
            corpus_config=corpus_config,
            tokens_transform_opts=args.tokens_transform_opts,
            extract_tagged_tokens_opts=args.extract_tagged_tokens_opts,
            tagged_tokens_filter_opts=args.tagged_tokens_filter_opts,
            vectorize_opts=args.vectorize_opts,
            checkpoint_filename=checkpoint_filename,
        ).value()

        if (args.count_threshold or 1) > 1:
            corpus = corpus.slice_by_n_count(args.count_threshold)

        if args.persist:
            store_corpus_bundle(corpus, args)

        return corpus

    except Exception as ex:
        raise ex
