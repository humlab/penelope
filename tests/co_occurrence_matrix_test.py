from typing import Iterator

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from penelope.utility import flatten, pretty_print_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from .utils import very_simple_corpus


def PPMI(C: scipy.sparse.csc_matrix) -> scipy.sparse.csc_matrix:
    """Tranform a counts matrix to PPMI.
        Source:
    Args:
      C: scipy.sparse.csc_matrix of counts C_ij

    Returns:
      (scipy.sparse.csc_matrix) PPMI(C) as defined above
    """
    # Total count.
    Z = float(C.sum())

    # Sum each row (along columns).
    Zr = np.array(C.sum(axis=1), dtype=np.float64).flatten()

    # Get indices of relevant elements.
    ii, jj = C.nonzero()  # row, column indices
    Cij = np.array(C[ii, jj], dtype=np.float64).flatten()

    # PMI equation.
    pmi = np.log(Cij * Z / (Zr[ii] * Zr[jj]))

    # Truncate to positive only.
    ppmi = np.maximum(0, pmi)  # take positive only

    # Re-format as sparse matrix.
    ret = scipy.sparse.csc_matrix((ppmi, (ii, jj)), shape=C.shape, dtype=np.float64)
    ret.eliminate_zeros()  # remove zeros
    return ret


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


def test_original():

    # Build a toy corpus with the same shape as our corpus object.
    toy_corpus = ["* nlp class is awesome *".split(), "nlp is awesome fun *".split()]

    vocab = set(flatten(toy_corpus))
    toy_tokens = flatten(toy_corpus)
    id_to_word = dict(enumerate(vocab))
    word_to_id = {v: k for k, v in id_to_word.items()}

    print(toy_tokens)
    toy_token_ids = [word_to_id[w] for w in toy_tokens]

    print(toy_tokens)
    V = len(vocab)

    toy_C = co_occurrence_matrix(toy_token_ids, V=V, K=1)

    ordered_vocab = [id_to_word[i] for i in range(0, len(vocab))]
    pretty_print_matrix(toy_C.toarray(), row_labels=ordered_vocab, column_labels=ordered_vocab, dtype=int)


def test_this_co_occurrence():

    corpus = very_simple_corpus(
        [
            ('rand_1991_5.txt', ['c', 'b', 'c', 'e', 'd', 'g', 'a']),
            ('rand_1991_6.txt', ['f', 'b', 'g', 'a', 'a']),
            ('rand_1993_7.txt', ['f', 'c', 'f', 'g']),
        ]
    )
    token2id = corpus.token2id
    token_ids = flatten([[token2id[w] for w in d] for d in corpus.terms])
    coo_matrix = co_occurrence_matrix(token_ids, V=len(token2id), K=2)

    assert coo_matrix is not None


# https://gist.github.com/zyocum/2ba0457246a4d0075149aa7d607432c1
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


class Cooccurrence(CountVectorizer):
    """Co-ocurrence matrix
    Convert collection of raw documents to word-word co-ocurrence matrix
    Parameters
    ----------
    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.
    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.
    max_df: float in range [0, 1] or int, default=1.0
    min_df: float in range [0, 1] or int, default=1
    Example
    -------
    >> import Cooccurrence
    >> docs = ['this book is good',
               'this cat is good',
               'cat is good shit']
    >> model = Cooccurrence()
    >> Xc = model.fit_transform(docs)
    Check vocabulary by printing
    >> model.vocabulary_
    """

    def __init__(
        self,
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        max_features=None,
        stop_words=None,
        normalize=True,
        vocabulary=None,
    ):

        super().__init__(
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            stop_words=stop_words,
            vocabulary=vocabulary,
        )

        self.normalize = normalize

    def fit_transform(self, raw_documents, y=None) -> csr_matrix:
        """Fit cooccurrence matrix
        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects
        Returns
        -------
        Xc : Cooccurrence matrix
        """
        X = super().fit_transform(raw_documents, y)

        Xc = X.T * X
        if self.normalize:
            g = sp.diags(1.0 / Xc.diagonal())
            Xc = g * Xc
        else:
            Xc.setdiag(0)

        return Xc
