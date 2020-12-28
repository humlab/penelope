from __future__ import annotations

import fnmatch
import re
from heapq import nlargest
from typing import Container, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn.preprocessing
from penelope import utility
from penelope.utility import getLogger, is_strictly_increasing
from sklearn.feature_extraction.text import TfidfTransformer

from .group import GroupByMixIn
from .interface import IVectorizedCorpus, VectorizedCorpusError
from .slice import SliceMixIn
from .store import StoreMixIn

# pylint: disable=logging-format-interpolation, too-many-public-methods

logger = getLogger("penelope")


class VectorizedCorpus(StoreMixIn, GroupByMixIn, SliceMixIn, IVectorizedCorpus):
    def __init__(
        self,
        bag_term_matrix: scipy.sparse.csr_matrix,
        token2id: Dict[str, int],
        document_index: pd.DataFrame,
        token_counter: Dict[str, int] = None,
    ):
        """Class that encapsulates a bag-of-word matrix.

        Parameters
        ----------
        bag_term_matrix : scipy.sparse.csr_matrix
            The bag-of-word matrix
        token2id : dict(str, int)
            Token to token id translation i.e. translates token to column index
        document_index : pd.DataFrame
            Corpus document metadata (bag-of-word row metadata)
        token_counter : dict(str,int), optional
            Total corpus word counts, by default None, computed if None
        """

        # Ensure that we have a sparse matrix (CSR)
        if not scipy.sparse.issparse(bag_term_matrix):
            bag_term_matrix = scipy.sparse.csr_matrix(bag_term_matrix)
        elif not scipy.sparse.isspmatrix_csr(bag_term_matrix):
            bag_term_matrix = bag_term_matrix.tocsr()

        self._bag_term_matrix: scipy.sparse.csr_matrix = bag_term_matrix

        assert scipy.sparse.issparse(self.bag_term_matrix), "only sparse data allowed"

        self._token2id = token2id
        self._id2token = None

        self._document_index = self._ingest_document_index(document_index=document_index)
        self._token_counter = token_counter

    def _ingest_document_index(self, document_index: pd.DataFrame):
        if not is_strictly_increasing(document_index.index):
            raise ValueError(
                "supplied `document index` must have an integer typed, strictly increasing index starting from 0"
            )
        if len(document_index) != self._bag_term_matrix.shape[0]:
            raise ValueError(
                f"expected `document index` to have length {self._bag_term_matrix.shape[0]} but found length {len(document_index)}"
            )

        if 'n_raw_tokens' not in document_index.columns:
            document_index['n_raw_tokens'] = self.document_token_counts

        return document_index

    @property
    def bag_term_matrix(self) -> scipy.sparse.csr_matrix:
        return self._bag_term_matrix

    @property
    def token2id(self) -> Dict[str, int]:
        return self._token2id

    @property
    def id2token(self) -> Mapping[int, str]:
        if self._id2token is None and self.token2id is not None:
            self._id2token = {i: t for t, i in self.token2id.items()}
        return self._id2token

    @property
    def vocabulary(self):
        vocab = [self.id2token[i] for i in range(0, self.data.shape[1])]
        return vocab

    @property
    def T(self) -> scipy.sparse.csr_matrix:
        """Returns transpose of BoW matrix """
        return self.bag_term_matrix.T

    @property
    def token_counter(self) -> Dict[str, int]:
        if self._token_counter is None:
            self._token_counter = {self.id2token[i]: c for i, c in enumerate(self.corpus_token_counts)}
        return self._token_counter

    @property
    def corpus_token_counts(self) -> np.ndarray:
        return self.bag_term_matrix.sum(axis=0).A1

    @property
    def document_token_counts(self) -> np.ndarray:
        return self.bag_term_matrix.sum(axis=1).A1

    @property
    def data(self) -> scipy.sparse.csr_matrix:
        """Returns BoW matrix """
        return self.bag_term_matrix

    @property
    def n_docs(self) -> int:
        """Returns number of documents """
        return self.bag_term_matrix.shape[0]

    @property
    def n_terms(self) -> int:
        """Returns number of types (unique words) """
        return self.bag_term_matrix.shape[1]

    @property
    def document_index(self) -> pd.DataFrame:
        """Returns number document index (part of interface) """
        return self._document_index

    def todense(self) -> VectorizedCorpus:
        """Returns dense BoW matrix"""
        dtm = self.data

        if scipy.sparse.issparse(dtm):
            dtm = dtm.todense()

        if isinstance(dtm, np.matrix):
            dtm = np.asarray(dtm)

        self._bag_term_matrix = dtm

        return self

    def get_word_vector(self, word: str):
        """Extracts vector (i.e. BoW matrix column for word's id) for word `word`

        Parameters
        ----------
        word : str

        Returns
        -------
        np.array
            BoW matrix column values found in column `token2id[word]`
        """
        return self.bag_term_matrix[:, self.token2id[word]].todense().A1  # x.A1 == np.asarray(x).ravel()

    def filter(self, px) -> VectorizedCorpus:
        """Returns a new corpus that only contains docs for which `px` is true.

        Parameters
        ----------
        px : Callable[Dict[str, Any], Boolean]
            The predicate that determines if document should be kept.

        Returns
        -------
        VectorizedCorpus
            Filtered corpus.
        """

        meta_documents = self.document_index[self.document_index.apply(px, axis=1)]

        indices = list(meta_documents.index)

        v_corpus = VectorizedCorpus(self.bag_term_matrix[indices, :], self.token2id, meta_documents, None)

        return v_corpus

    def normalize(self, axis: int = 1, norm: str = 'l1', keep_magnitude: bool = False) -> IVectorizedCorpus:
        """Scale BoW matrix's rows or columns individually to unit norm:

            sklearn.preprocessing.normalize(self.bag_term_matrix, axis=axis, norm=norm)

        Parameters
        ----------
        axis : int, optional
            Axis used to normalize the data along. 1 normalizes each row (bag/document), 0 normalizes each column (word).
        norm : str, optional
            Norm to use 'l1', 'l2', or 'max' , by default 'l1'
        keep_magnitude : bool, optional
            Scales result matrix so that sum equals input matrix sum, by default False

        Returns
        -------
        VectorizedCorpus
            New corpus normalized in given `axis`
        """
        btm = sklearn.preprocessing.normalize(self.bag_term_matrix, axis=axis, norm=norm)

        if keep_magnitude is True:
            factor = self.bag_term_matrix[0, :].sum() / btm[0, :].sum()
            btm = btm * factor

        corpus = VectorizedCorpus(btm, self.token2id, self.document_index, self.token_counter)

        return corpus

    def normalize_by_raw_counts(self):

        if 'n_raw_tokens' not in self.document_index.columns:
            # logging.warning("Normalizing using DTM counts (not actual self counts)")
            # return self.normalize()
            raise VectorizedCorpusError("raw count normalize attempted but no n_raw_tokens in document index")

        token_counts = self.document_index.n_raw_tokens.values
        btm = utility.normalize_sparse_matrix_by_vector(self.bag_term_matrix, token_counts)
        corpus = VectorizedCorpus(btm, self.token2id, self.document_index, self.token_counter)

        return corpus

    def n_top_tokens(self, n_top) -> Dict[str, int]:
        """Returns `n_top` most frequent words.

        Parameters
        ----------
        n_top : int
            Number of words to return

        Returns
        -------
        Dict[str, int]
            Most frequent words and their counts, subset of dict `token_counter`

        """
        tokens = {w: self.token_counter[w] for w in nlargest(n_top, self.token_counter, key=self.token_counter.get)}
        return tokens

    def stats(self):
        """Returns (and prints) some corpus status
        Returns
        -------
        dict
            Corpus stats
        """
        stats_data = {
            'bags': self.bag_term_matrix.shape[0],
            'vocabulay_size': self.bag_term_matrix.shape[1],
            'sum_over_bags': self.bag_term_matrix.sum(),
            '10_top_tokens': ' '.join(self.n_top_tokens(10).keys()),
        }
        for key in stats_data:
            logger.info('   {}: {}'.format(key, stats_data[key]))
        return stats_data

    def to_n_top_dataframe(self, n_top: int) -> pd.DataFrame:
        """Returns BoW as a Pandas dataframe with the `n_top` most common words.

        Parameters
        ----------
        n_top : int
            Number of top words to return.

        Returns
        -------
        DataFrame
            BoW for top `n_top` words
        """
        v_n_corpus = self.slice_by_n_top(n_top)
        data = v_n_corpus.bag_term_matrix.T
        df = pd.DataFrame(
            data=data.todense(),
            index=[v_n_corpus.id2token[i] for i in range(0, n_top)],
            columns=range(0, v_n_corpus.n_docs),
        )
        return df

    def year_range(self) -> Tuple[Optional[int], Optional[int]]:
        """Returns document's year range

        Returns
        -------
        Tuple[Optional[int],Optional[int]]
            Min/max document year
        """
        if 'year' in self.document_index.columns:
            return (self.document_index.year.min(), self.document_index.year.max())
        return (None, None)

    def xs_years(self) -> Tuple[int, int]:
        """Returns an array that contains a no-gap year sequence from min year to max year

        Returns
        -------
        numpy.array
            Sequence from min year to max year
        """
        (low, high) = self.year_range()
        xs = np.arange(low, high + 1, 1)
        return xs

    def token_indices(self, tokens: Iterable[str]):
        """Returns token (column) indices for words `tokens`

        Parameters
        ----------
        tokens : list(str)
            Input words

        Returns
        -------
        Iterable[str]
            Input words' column indices in the BoW matrix
        """
        return [self.token2id[token] for token in tokens]

    def tf_idf(self, norm: str = 'l2', use_idf: bool = True, smooth_idf: bool = True) -> IVectorizedCorpus:
        """Returns a (nomalized) TF-IDF transformed version of the corpus

        Calls sklearn's TfidfTransformer

        Parameters
        ----------
        norm : str, optional
            Specifies row unit norm, `l1` or `l2`, default 'l2'
        use_idf : bool, default True
            Indicates if an IDF reweighting should be done
        smooth_idf : bool, optional
            Adds 1 to document frequencies to smooth the IDF weights, by default True

        Returns
        -------
        VectorizedCorpus
            The TF-IDF transformed corpus
        """
        transformer = TfidfTransformer(norm=norm, use_idf=use_idf, smooth_idf=smooth_idf)

        tfidf_bag_term_matrix = transformer.fit_transform(self.bag_term_matrix)

        n_corpus = VectorizedCorpus(tfidf_bag_term_matrix, self.token2id, self.document_index, self.token_counter)

        return n_corpus

    def to_bag_of_terms(self, indicies: Optional[Iterable[int]] = None) -> Iterable[Iterable[str]]:
        """Returns a document token stream that corresponds to the BoW.
        Tokens are repeated according to BoW token counts.
        Note: Will not work on a normalized corpus!

        Parameters
        ----------
        indicies : Optional[Iterable[int]], optional
            Specifies word subset, by default None

        Returns
        -------
        Iterable[Iterable[str]]
            Documenttoken stream.
        """
        dtm = self.bag_term_matrix
        indicies = indicies or range(0, dtm.shape[0])
        id2token = self.id2token
        return (
            (w for ws in (dtm[doc_id, i] * [id2token[i]] for i in dtm[doc_id, :].nonzero()[1]) for w in ws)
            for doc_id in indicies
        )

    def get_top_n_words(self, n=1000, indices=None):
        """Returns the top n words in a subset of the corpus sorted according to occurrence. """
        if indices is None:
            sum_words = self.bag_term_matrix.sum(axis=0)
        else:
            sum_words = self.bag_term_matrix[indices, :].sum(axis=0)

        id2token = self.id2token
        token_ids = sum_words.nonzero()[1]
        words_freq = [(id2token[i], sum_words[0, i]) for i in token_ids]

        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

        return words_freq[:n]

    def co_occurrence_matrix(self) -> scipy.sparse.spmatrix:
        """Computes (document) cooccurence matrix

        Returns
        -------
        Tuple[scipy.sparce.spmatrix. Dict[int,str]]
            The co-occurrence matrix
        """
        term_term_matrix = np.dot(self.bag_term_matrix.T, self.bag_term_matrix)
        term_term_matrix = scipy.sparse.triu(term_term_matrix, 1)

        return term_term_matrix

    def find_matching_words(self, word_or_regexp: List[str], n_max_count: int, descending: bool = False) -> List[str]:
        """Returns words in corpus that matches candidate tokens """
        words = self.pick_n_top_words(
            find_matching_words_in_vocabulary(self.token2id, word_or_regexp),
            n_max_count,
            descending=descending,
        )
        return words

    def find_matching_words_indices(self, word_or_regexp: List[str], n_max_count: int, descending: bool = False) -> List[int]:
        """Returns `tokens´ indicies` in corpus that matches candidate tokens """
        indices: List[int] = [
            self.token2id[token]
            for token in self.find_matching_words(word_or_regexp, n_max_count, descending=descending)
            if token in self.token2id
        ]
        return indices

    def pick_n_top_words(self, words: Container[str], n_top: int, descending: bool = False) -> List[str]:
        """Returns the `n_top` most frequent word in `tokens`"""
        words = list(words)
        if len(words) < n_top:
            return words
        token_counts = [self.token_counter.get(w, 0) for w in words]
        most_frequent_words = [words[x] for x in np.argsort(token_counts)[-n_top:]]
        if descending:
            most_frequent_words = list(sorted(most_frequent_words, reverse=descending))
        return most_frequent_words

    @staticmethod
    def create(
        bag_term_matrix: scipy.sparse.csr_matrix,
        token2id: Dict[str, int],
        document_index: pd.DataFrame,
        token_counter: Dict[str, int] = None,
    ) -> "IVectorizedCorpus":
        return VectorizedCorpus(
            bag_term_matrix=bag_term_matrix,
            token2id=token2id,
            document_index=document_index,
            token_counter=token_counter,
        )


def find_matching_words_in_vocabulary(token2id: Container[str], candidate_words: Set[str]) -> Set[str]:

    words = {w for w in candidate_words if w in token2id}

    remaining_words = [w for w in candidate_words if w not in words and len(w) > 0]

    word_exprs = [x for x in remaining_words if "*" in x or (x.startswith("|") and x.endswith("|"))]

    for expr in word_exprs:

        if expr.startswith("|") and expr.endswith("|"):
            pattern = re.compile(expr.strip('|'))  # "^.*tion$"
            words |= {x for x in token2id if x not in words and pattern.match(x)}
        else:
            words |= {x for x in token2id if x not in words and fnmatch.fnmatch(x, expr)}

    return words
