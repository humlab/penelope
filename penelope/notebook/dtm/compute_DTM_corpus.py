from typing import Callable

import penelope.corpus as corpora
import penelope.corpus.dtm as dtm
import penelope.pipeline as pipeline
from penelope.corpus.dtm.vectorized_corpus import VectorizedCorpus

from ..interface import ComputeOpts
from . import factory as corpus_factory
from .store import store_corpus_bundle


def compute_document_term_matrix(
    corpus_config: pipeline.CorpusConfig,  # pylint: disable=unused-argument
    pipeline_factory: Callable,  # pylint: disable=unused-argument
    args: ComputeOpts,
) -> dtm.VectorizedCorpus:

    try:

        assert args.is_satisfied()

        tokenized_corpus: corpora.TokenizedCorpus = corpus_factory.create_corpus(
            corpus_type=args.corpus_type,
            input_filename=args.corpus_filename,
            tokens_transform_opts=args.tokens_transform_opts,
            extract_tokens_opts=args.extract_tagged_tokens_opts,
            reader_opts=args.text_reader_opts,
        )
        corpus: VectorizedCorpus = dtm.CorpusVectorizer().fit_transform(tokenized_corpus, **args.vectorize_opts.props)

        if (args.count_threshold or 1) > 1:
            corpus = corpus.slice_by_n_count(args.count_threshold)

        if args.persist:
            store_corpus_bundle(corpus, args)

        return corpus

    except Exception as ex:
        raise ex
