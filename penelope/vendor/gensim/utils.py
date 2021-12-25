from __future__ import annotations

import collections
from typing import Iterable

from gensim.corpora.dictionary import Dictionary
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
