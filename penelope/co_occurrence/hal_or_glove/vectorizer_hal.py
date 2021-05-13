import itertools
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import penelope.utility as utility
import penelope.vendor.gensim as gensim_utility
import scipy.sparse as sp

logger = utility.getLogger('corpus_text_analysis')


# pylint: disable=too-many-instance-attributes
class HyperspaceAnalogueToLanguageVectorizer:
    def __init__(self, tokens_stream: Iterable[Iterable[str]] = None, token2id: Mapping[str, int] = None):
        """
        Build vocabulary and create nw_xy term-term matrix and nw_x term global occurence vector

        Parameter:
            corpus Iterable[Iterable[str]]

        """
        self.token2id: Mapping[str, int] = token2id
        self._id2token: Mapping[int, str] = None
        self.corpus: Iterable[Iterable[str]] = tokens_stream

        self.nw_xy: sp.csc_matrix = None
        self.nw_x: np.ndarray = None

    @property
    def corpus(self) -> Iterable[Iterable[str]]:
        return self._corpus

    @corpus.setter
    def corpus(self, value: Iterable[Iterable[str]]):

        self._corpus = value
        self.term_count = sum(map(len, value or []))

        if self.token2id is None and value is not None:
            self.token2id = gensim_utility.build_vocab(value)
            self._id2token = None

    @property
    def id2token(self) -> Mapping[int, str]:
        if self._id2token is None:
            if self.token2id is not None:
                self._id2token = {v: k for k, v in self.token2id.items()}
        return self._id2token

    def sliding_window(self, seq: Iterable[str], n: int) -> Iterable[str]:
        it: Iterable[str] = itertools.chain(iter(seq), [None] * n)
        memory: Iterable[str] = tuple(itertools.islice(it, n + 1))
        if len(memory) == n + 1:
            yield memory
        for x in it:
            memory = memory[1:] + (x,)
            yield memory

    def fit(
        self, corpus: Iterable[str] = None, size: int = 2, distance_metric: int = 0, zero_out_diag: bool = False
    ) -> "HyperspaceAnalogueToLanguageVectorizer":
        '''Trains HAL for a document. Note that sentence borders (for now) are ignored'''

        if corpus is not None:
            self.corpus = corpus

        assert self.token2id is not None, "Fit with no vocabulary!"
        assert self.corpus is not None, "Fit with no corpus!"

        nw_xy: sp.lil_matrix = sp.csc_matrix((len(self.token2id), len(self.token2id)), dtype=np.uint32)
        nw_x: np.ndarray = np.zeros(len(self.token2id), dtype=np.uint32)

        for terms in corpus:

            id_terms = (self.token2id[t] for t in terms)

            for win in self.sliding_window(id_terms, size):

                # logger.info([ self.id2token[x] if x is not None else None for x in win])

                if win[0] is None:
                    continue

                for x in win:
                    if x is not None:
                        nw_x[x] += 1

                for i in range(1, size + 1):

                    if win[i] is None:
                        continue

                    if zero_out_diag:
                        if win[0] == win[i]:
                            continue

                    d = float(i)  # abs(n - i)
                    if distance_metric == 0:  # linear i.e. adjacent equals window size, then decreasing by one
                        w = size - d + 1  # / size
                    elif distance_metric == 1:  # f(d) = 1 / d
                        w = 1.0 / d
                    elif distance_metric == 2:  # Constant value of 1
                        w = 1
                    else:
                        w = 0

                    # print('*', i, self.id2token[win[0]], self.id2token[win[i]], w, [ self.id2token[x] if x is not None else None for x in win])
                    nw_xy[win[0], win[i]] += w

        self.nw_x = nw_x
        self.nw_xy = nw_xy
        # self.f_xy = nw_xy / np.max(nw_xy)

        return self

    def to_df(self) -> pd.DataFrame:
        columns = [self.id2token[i] for i in range(0, len(self.token2id))]
        return pd.DataFrame(data=self.nw_xy.todense(), index=list(columns), columns=list(columns), dtype=np.float64).T

    # def xxx_cwr(self, direction_sensitive=False, normalize='size'):

    #     n = self.nw_x.shape[0]

    #     nw = self.nw_x.reshape(n,1)
    #     nw_xy = self.nw_xy

    #     norm = 1.0
    #     if normalize == 'size':
    #         norm = float(self.term_count)
    #     elif norm == 'max':
    #         norm = float(np.max(nw_xy))
    #     elif norm == 'sum':
    #         norm = float(np.sum(nw_xy))

    #     # nw.resize(nw.shape[0], 1)

    #     self.cwr = sp.lil_matrix(nw_xy / (-nw_xy + nw + nw.T)) # nw.reshape(n,1).T))

    #     if norm != 1.0:
    #         self.cwr = self.cwr / norm

    #     coo_matrix = self.cwr.tocoo(copy=False)
    #     df = pd.DataFrame({
    #         'x_id': coo_matrix.row,
    #         'y_id': coo_matrix.col,
    #         'cwr': coo_matrix.data
    #     }).sort_values(['x_id', 'y_id']).reset_index(drop=True)

    #     df = df.assign(
    #         x_term=df.x_id.apply(lambda x: self.id2token[x]),
    #         y_term=df.y_id.apply(lambda x: self.id2token[x])
    #     )

    #     df_nw_x = pd.DataFrame(self.nw_x, columns=['nw'])

    #     df = df.merge(df_nw_x, left_on='x_id', right_index=True, how='inner').rename(columns={'nw': 'nw_x'})
    #     df = df.merge(df_nw_x, left_on='y_id', right_index=True, how='inner').rename(columns={'nw': 'nw_y'})

    #     df = df[['x_id', 'y_id', 'x_term', 'y_term', 'cwr']]

    #     return df

    # def cooccurence2(self, direction_sensitive=False, normalize='size', zero_diagonal=True):
    #     n = self.cwr.shape[0]
    #     df = pd.DataFrame([(
    #             i,
    #             j,
    #             self.id2token[i],
    #             self.id2token[j],
    #             self.nw_xy[i,j],
    #             self.nw_x[i],
    #             self.nw_x[j],
    #             self.cwr[i,j]
    #         ) for i,j in itertools.product(range(0,n), repeat=2) if self.cwr[i,j] > 0 ], columns=['x_id', 'y_id', 'x_term', 'y_term', 'nw_xy', 'nw_x', 'nw_y', 'cwr'])

    #     return df

    def to_dataframe(
        self, *, normalize: str = 'size', zero_diagonal: bool = True, direction_sensitive: bool = False, **_
    ) -> pd.DataFrame:
        '''Return computed co-occurrence values'''

        matrix: sp.csc_matrix = self.nw_xy

        if not direction_sensitive:
            matrix += matrix.T
            matrix[np.tril_indices(matrix.shape[0])] = 0
        else:
            if zero_diagonal:
                matrix.fill_diagonal(0)

        coo_matrix: sp.coo_matrix = matrix.tocoo(copy=False)

        df_nw_x: pd.DataFrame = pd.DataFrame(self.nw_x, columns=['nw'])

        df: pd.DataFrame = (
            pd.DataFrame({'x_id': coo_matrix.row, 'y_id': coo_matrix.col, 'nw_xy': coo_matrix.data})[
                ['x_id', 'y_id', 'nw_xy']
            ]
            .sort_values(['x_id', 'y_id'])
            .reset_index(drop=True)
        )

        df = df.assign(
            x_term=df.x_id.apply(lambda x: self.id2token[x]), y_term=df.y_id.apply(lambda x: self.id2token[x])
        )

        df = df.merge(df_nw_x, left_on='x_id', right_index=True, how='inner').rename(columns={'nw': 'nw_x'})
        df = df.merge(df_nw_x, left_on='y_id', right_index=True, how='inner').rename(columns={'nw': 'nw_y'})

        df = df[['x_id', 'y_id', 'x_term', 'y_term', 'nw_xy', 'nw_x', 'nw_y']]

        norm = 1.0
        if normalize == 'size':
            norm = self.term_count
        elif normalize == 'max':
            norm = np.max(coo_matrix)
        elif normalize is None:
            logger.warning('No normalize method specified. Using absolute counts...')
            # return as as is..."
        else:
            assert False, 'Unknown normalize specifier'

        # logger.info('Normalizing for document corpus size %s.', norm)

        df_nw_xy: pd.DataFrame = df.assign(cwr=((df.nw_xy / (df.nw_x + df.nw_y - df.nw_xy)) / norm))

        df_nw_xy.loc[df_nw_xy.cwr < 0.0, 'cwr'] = 0
        df_nw_xy.cwr.fillna(0.0, inplace=True)

        return df_nw_xy[df_nw_xy.cwr > 0]


def test_burgess_litmus_test():
    terms = 'The Horse Raced Past The Barn Fell .'.lower().split()
    answer = {
        'barn': {'.': 4, 'barn': 0, 'fell': 5, 'horse': 0, 'past': 0, 'raced': 0, 'the': 0},
        'fell': {'.': 5, 'barn': 0, 'fell': 0, 'horse': 0, 'past': 0, 'raced': 0, 'the': 0},
        'horse': {'.': 0, 'barn': 2, 'fell': 1, 'horse': 0, 'past': 4, 'raced': 5, 'the': 3},
        'past': {'.': 2, 'barn': 4, 'fell': 3, 'horse': 0, 'past': 0, 'raced': 0, 'the': 5},
        'raced': {'.': 1, 'barn': 3, 'fell': 2, 'horse': 0, 'past': 5, 'raced': 0, 'the': 4},
        'the': {'.': 3, 'barn': 6, 'fell': 4, 'horse': 5, 'past': 3, 'raced': 4, 'the': 2},
    }
    df_answer = pd.DataFrame(answer).astype(np.uint32)[['the', 'horse', 'raced', 'past', 'barn', 'fell']].sort_index()
    # display(df_answer)
    vectorizer = HyperspaceAnalogueToLanguageVectorizer()
    vectorizer.fit([terms], size=5, distance_metric=0)
    df_imp = vectorizer.to_df().astype(np.uint32)[['the', 'horse', 'raced', 'past', 'barn', 'fell']].sort_index()
    assert df_imp.equals(df_answer), "Test failed"
    # df_imp == df_answer

    # Example in Chen, Lu:
    terms = 'The basic concept of the word association'.lower().split()
    vectorizer = HyperspaceAnalogueToLanguageVectorizer().fit([terms], size=5, distance_metric=0)
    df_imp = vectorizer.to_df().astype(np.uint32)[['the', 'basic', 'concept', 'of', 'word', 'association']].sort_index()
    df_answer = pd.DataFrame(
        {
            'the': [2, 5, 4, 3, 6, 4],
            'basic': [3, 0, 5, 4, 2, 1],
            'concept': [4, 0, 0, 5, 3, 2],
            'of': [5, 0, 0, 0, 4, 3],
            'word': [0, 0, 0, 0, 0, 5],
            'association': [0, 0, 0, 0, 0, 0],
        },
        index=['the', 'basic', 'concept', 'of', 'word', 'association'],
        dtype=np.uint32,
    ).sort_index()[['the', 'basic', 'concept', 'of', 'word', 'association']]
    assert df_imp.equals(df_answer), "Test failed"
    print('Test run OK')


# test_burgess_litmus_test()
