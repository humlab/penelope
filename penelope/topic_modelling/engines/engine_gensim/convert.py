from __future__ import annotations

from typing import Any, Mapping, Tuple, Union

import scipy.sparse as sp
from gensim.corpora.dictionary import Dictionary
from gensim.matutils import Sparse2Corpus
from loguru import logger
from penelope import corpus as pc
from penelope.vendor import gensim as gs

# pylint: disable=unused-argument


class Id2TokenMissingError(NotImplementedError):
    ...


class TranslateCorpus:
    def translate(self, source: Any, *, id2token: Mapping[int, str] = None) -> Tuple[Sparse2Corpus, Dictionary]:

        """Gensim doc says:
        "corpus : iterable of list of (int, float), optional
        Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).
        If you have a CSC in-memory matrix, you can convert it to a
        streamed corpus with the help of gensim.matutils.Sparse2Corpus.
        If not given, the model is left untrained ...."
        """
        vocabulary: Union[Dictionary, Mapping[str, int]] = None
        corpus: Sparse2Corpus = None

        if isinstance(source, Sparse2Corpus):
            corpus = source
        elif isinstance(source, pc.VectorizedCorpus):
            corpus = Sparse2Corpus(source.data, documents_columns=False)
        elif sp.issparse(source):
            corpus = Sparse2Corpus(source, documents_columns=False)
        else:
            """Assumes stream of (document, tokens)"""
            vocabulary = (
                gs.from_stream_of_tokens_to_dictionary(source, id2token)
                if id2token is None
                else pc.id2token2token2id(id2token)
            )
            corpus = gs.from_stream_of_tokens_to_sparse2corpus(source, vocabulary)

        if vocabulary is None:
            """Build from corpus, `id2token` must be supplied"""
            if id2token is None:
                raise Id2TokenMissingError()
            vocabulary: dict = (
                Dictionary.from_corpus(pc.csr2bow(corpus.sparse), id2word=id2token)
                if id2token is None
                else pc.id2token2token2id(id2token)
            )

        if id2token is not None:
            logger.warning("skipping build of `Dictionary` (using existing dict)")
        return corpus, vocabulary
