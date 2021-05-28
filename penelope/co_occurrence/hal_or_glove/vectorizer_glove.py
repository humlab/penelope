import numpy as np
import pandas as pd
import penelope.vendor.gensim as gensim_utility
from loguru import logger

try:
    import glove  # type: ignore # pylint: disable=import-error
except ModuleNotFoundError:
    glove = None


# See http://www.foldl.me/2014/glove-python/
class GloveVectorizer:
    def __init__(self, corpus=None, token2id=None):

        assert glove is not None, "Glove is not installed!"

        self.token2id = token2id
        self._id2token = None
        self.corpus = corpus
        self.nw_xy = None

    @property
    def corpus(self):
        return self._corpus

    @corpus.setter
    def corpus(self, value):

        self._corpus = value
        self.term_count = sum(map(len, value or []))

        if self.token2id is None and value is not None:
            self.token2id = gensim_utility.build_vocab(value)
            self._id2token = None

    @property
    def id2token(self):
        if self._id2token is None:
            if self.token2id is not None:
                self._id2token = {v: k for k, v in self.token2id.items()}
        return self._id2token

    # def fit(self, sentences, window=2, dictionary=None):
    def fit(self, corpus=None, size=2):  # , distance_metric=0, zero_out_diag=False):

        if corpus is not None:
            self.corpus = corpus

        assert self.token2id is not None, "Fit with no vocabulary!"
        assert self.corpus is not None, "Fit with no corpus!"

        glove_corpus = glove.Corpus(dictionary=self.token2id)
        glove_corpus.fit(corpus, window=size)

        self.nw_xy = glove_corpus.matrix

        return self

    def to_dataframe(self, *, normalize='size', zero_diagonal=True, **_):
        '''Return computed co-occurrence values'''

        matrix = self.nw_xy

        if zero_diagonal:
            pass
        #    matrix.fill_diagonal(0)

        coo_matrix = matrix  # .tocoo(copy=False)

        df = pd.DataFrame(
            {
                'x_id': coo_matrix.row,
                'y_id': coo_matrix.col,
                'nw_xy': coo_matrix.data,
                'nw_x': 0,
                'nw_y': 0,
            }
        ).reset_index(drop=True)

        df = df.assign(
            x_term=df.x_id.apply(lambda x: self.id2token[x]), y_term=df.y_id.apply(lambda x: self.id2token[x])
        )

        df = df[['x_id', 'y_id', 'x_term', 'y_term', 'nw_xy', 'nw_x', 'nw_y']]

        norm = 1.0
        if normalize == 'size':
            norm = self.term_count
        elif normalize == 'max':
            norm = np.max(coo_matrix)
        elif normalize is None:
            logger.warning('No normalize method specified. Using absolute counts...')
        else:
            assert False, 'Unknown normalize specifier'

        df_nw_xy = df.assign(cwr=df.nw_xy / norm)

        return df_nw_xy[df_nw_xy.cwr > 0]
