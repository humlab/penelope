from typing import Iterator

import numpy as np
import pandas as pd
import scipy

from penelope.corpus import TokenizedCorpus
from penelope.corpus.readers import InMemoryReader


def co_occurrence_matrix(token_ids: Iterator[int], V: int, K: int = 2) -> scipy.sparse.spmatrix:
    """Computes a sparse co-occurrence matrix given a corpus

    Source: https://colab.research.google.com/github/henrywoo/MyML/blob/master/Copy_of_nlu_2.ipynb#scrollTo=hPySe-BBEVRy

    Parameters
    ----------
    token_ids : Iterator[int]
        Corpus token ids
    V : int
        A vocabulary size V
    K : int, optional
        K (the context window is +-K)., by default 2

    Returns
    -------
    scipy.sparse.spmatrix
        Sparse co-occurrence matrix
    """
    C = scipy.sparse.csc_matrix((V, V), dtype=np.float32)

    for k in range(1, K + 1):
        print(f'Counting pairs (i, i | {k}) ...')
        i = token_ids[:-k]  # current word
        j = token_ids[k:]  # k words ahead
        data = (np.ones_like(i), (i, j))  # values, indices
        Ck_plus = scipy.sparse.coo_matrix(data, shape=C.shape, dtype=np.float32)
        Ck_plus = scipy.sparse.csc_matrix(Ck_plus)
        Ck_minus = Ck_plus.T  # consider k words behind
        C += Ck_plus + Ck_minus

    print(f"Co-occurrence matrix: {C.shape[0]} words x {C.shape[0]} words")
    print(f" {C.nnz} nonzero elements")
    return C


def test_this_co_occurrence():

    SIMPLE_CORPUS_ABCDEFG_7DOCS = [
        ('rand_1991_1.txt', ['b', 'd', 'a', 'c', 'e', 'b', 'a', 'd', 'b']),
        ('rand_1992_2.txt', ['b', 'f', 'e', 'e', 'f', 'e', 'a', 'a', 'b']),
        ('rand_1992_3.txt', ['a', 'e', 'f', 'b', 'e', 'a', 'b', 'f']),
        ('rand_1992_4.txt', ['e', 'a', 'a', 'b', 'g', 'f', 'g', 'b', 'c']),
        ('rand_1991_5.txt', ['c', 'b', 'c', 'e', 'd', 'g', 'a']),
        ('rand_1991_6.txt', ['f', 'b', 'g', 'a', 'a']),
        ('rand_1993_7.txt', ['f', 'c', 'f', 'g']),
    ]

    reader = InMemoryReader(SIMPLE_CORPUS_ABCDEFG_7DOCS, filename_fields="year:_:1")
    corpus = TokenizedCorpus(reader=reader)

    vocabulary = corpus.token2id

    token_ids = [vocabulary[w] for w in SIMPLE_CORPUS_ABCDEFG_7DOCS[0][1]]
    m = co_occurrence_matrix(token_ids=token_ids, V=len(vocabulary), K=2)

    assert m is not None


# https://gist.github.com/zyocum/2ba0457246a4d0075149aa7d607432c1

# Refrence are teaken form
# https://www.kaggle.com/ambarish/recommendation-system-donors-choose
# https://github.com/roothd17/Donor-Choose-ML
# https://github.com/harrismohammed?tab=repositories


def COmatrix(data, words, cw=5):

    cm = pd.DataFrame(np.zeros((len(words), len(words))), index=words, columns=words)

    for sent in data['combined']:

        word = sent.split()

        for ind in range(len(word)):

            if cm.get(word[ind]) is None:
                continue

            for i in range(1, cw + 1):

                if ind - i >= 0:
                    if cm.get(word[ind - i]) is not None:

                        cm[word[ind - i]].loc[word[ind]] = cm.get(word[ind - i]).loc[word[ind]] + 1
                        cm[word[ind]].loc[word[ind - i]] = cm.get(word[ind]).loc[word[ind - i]] + 1

                if ind + i < len(word):
                    if cm.get(word[ind + i]) is not None:

                        cm[word[ind + i]].loc[word[ind]] = cm.get(word[ind + i]).loc[word[ind]] + 1
                        cm[word[ind]].loc[word[ind + i]] = cm.get(word[ind]).loc[word[ind + i]] + 1

    np.fill_diagonal(cm.values, 0)
    cm = cm.div(2)

    return cm
