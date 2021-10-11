from __future__ import annotations

import collections
from typing import Iterable, Tuple

from gensim.corpora.dictionary import Dictionary
from gensim.matutils import Sparse2Corpus, corpus2csc
from loguru import logger


def build_vocab(corpus: Iterable[Iterable[str]]) -> dict:
    ''' Iterates corpus and add distinct terms to vocabulary '''
    logger.info('Builiding vocabulary...')
    token2id = collections.defaultdict()
    token2id.default_factory = token2id.__len__
    for doc in corpus:
        for term in doc:
            token2id[term]  # pylint: disable=pointless-statement
    logger.info('Vocabulary of size {} built.'.format(len(token2id)))
    return token2id


def create_dictionary(id2word: dict) -> Dictionary:

    if isinstance(id2word, Dictionary):
        return id2word

    if not isinstance(id2word, dict):
        raise ValueError(f"expected dict, found {type(id2word)}")

    dictionary: Dictionary = Dictionary()
    dictionary.id2token = id2word
    dictionary.token2id = dict((v, k) for v, k in id2word.items())

    return dictionary


def terms_to_sparse_corpus(source: Iterable[Iterable[str]]) -> Tuple[Sparse2Corpus, Dictionary]:
    """Convert stream of (stream of) tokens to a Gensim sparse corpus"""

    id2word: Dictionary = Dictionary(source)
    bow_corpus: Iterable[Iterable[int | float]] = [id2word.doc2bow(tokens) for tokens in source]
    csc_matrix = corpus2csc(
        bow_corpus,
        num_terms=len(id2word),
        num_docs=len(bow_corpus),
        num_nnz=sum(map(len, bow_corpus)),
    )
    corpus: Sparse2Corpus = Sparse2Corpus(csc_matrix, documents_columns=True)
    return corpus, id2word
