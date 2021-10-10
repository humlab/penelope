from __future__ import annotations

import contextlib
import fnmatch
import re
import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import scipy
import sklearn.preprocessing
from penelope import utility

# pylint: disable=logging-format-interpolation, too-many-public-methods, too-many-ancestors
from scipy.sparse import SparseEfficiencyWarning
from sklearn.feature_extraction.text import TfidfTransformer

from ..document_index import DocumentIndex
from .group import GroupByMixIn
from .interface import IVectorizedCorpus, VectorizedCorpusError
from .slice import SliceMixIn
from .stats import StatsMixIn
from .store import StoreMixIn
from .ttm import CoOccurrenceMixIn

warnings.simplefilter('ignore', SparseEfficiencyWarning)

# FIXME #109 Refactor VectorizedCorpus to use Token2Id?
class VectorizedCorpus(StoreMixIn, GroupByMixIn, SliceMixIn, StatsMixIn, CoOccurrenceMixIn, IVectorizedCorpus):
    def __init__(
        self,
        bag_term_matrix: scipy.sparse.csr_matrix,
        token2id: Dict[str, int],
        document_index: DocumentIndex,
        overridden_term_frequency: Optional[Dict[int, int]] = None,
        **kwargs,
    ):
        """Class that encapsulates a bag-of-word matrix

        Args:
            bag_term_matrix (scipy.sparse.csr_matrix): Bag-of-word matrix
            token2id (Dict[str, int]): Token to token/column index translation
            document_index (DocumentIndex): Corpus document/row metadata
            overridden_term_frequency (np.ndarrys, optional): Supplied if source TF
        """
        self._class_name: str = "penelope.corpus.dtm.corpus.VectorizedCorpus"

        # Ensure that we have a sparse matrix (CSR)
        if not scipy.sparse.issparse(bag_term_matrix):
            bag_term_matrix = scipy.sparse.csr_matrix(bag_term_matrix)
        elif not scipy.sparse.isspmatrix_csr(bag_term_matrix):
            bag_term_matrix = bag_term_matrix.tocsr()

        self._bag_term_matrix: scipy.sparse.csr_matrix = bag_term_matrix
        self._token2id: Mapping[str, int] = token2id
        self._id2token: Optional[Mapping[int, str]] = None
        self._document_index: DocumentIndex = self._ingest_document_index(document_index=document_index)
        self._overridden_term_frequency: Optional[np.ndarray] = overridden_term_frequency
        self._payload: dict = dict(**kwargs)

    def _ingest_document_index(self, document_index: DocumentIndex):
        if not utility.is_strictly_increasing(document_index.index):
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
    def vocabulary(self) -> List[str]:
        vocab = [self.id2token[i] for i in range(0, self.data.shape[1])]
        return vocab

    @property
    def T(self) -> scipy.sparse.csr_matrix:
        """Returns transpose of BoW matrix """
        return self._bag_term_matrix.T

    @property
    def term_frequency0(self) -> np.ndarray:
        """Global TF (absolute term count), overridden prioritized"""
        if self._overridden_term_frequency is not None:
            return self._overridden_term_frequency
        return self.term_frequency

    @property
    def term_frequency(self) -> np.ndarray:
        """Global TF (absolute term count)"""
        return self._bag_term_matrix.sum(axis=0).A1.ravel()

    @property
    def overridden_term_frequency(self) -> np.ndarray:
        """Overridden global token frequencies (source corpus size)"""
        return self._overridden_term_frequency

    def term_frequency_map(self) -> Mapping[str, int]:
        fg = self.id2token.get
        tf = self.term_frequency
        return {fg(i): tf[i] for i in range(0, len(self.token2id))}

    @property
    def TF(self) -> np.ndarray:
        """Term frequencies (TF)"""
        return self.term_frequency

    @property
    def document_token_counts(self) -> np.ndarray:
        """Number of tokens per document"""
        return self._bag_term_matrix.sum(axis=1).A1

    @property
    def data(self) -> scipy.sparse.csr_matrix:
        """Returns BoW matrix """
        return self._bag_term_matrix

    @property
    def shape(self) -> Tuple[int, int]:
        return self._bag_term_matrix.shape

    @property
    def n_docs(self) -> int:
        """Returns number of documents """
        return self._bag_term_matrix.shape[0]

    @property
    def n_terms(self) -> int:
        """Returns number of types (unique words) """
        return self._bag_term_matrix.shape[1]

    @property
    def document_index(self) -> DocumentIndex:
        """Returns number document index (part of interface) """
        return self._document_index

    @property
    def payload(self) -> Mapping[Any, Any]:
        return self._payload

    def remember(self, **kwargs) -> VectorizedCorpus:
        """Stores items in payload"""
        self.payload.update(kwargs)
        return self

    def recall(self, key: str) -> Optional[Any]:
        """Retrieves item from payload"""
        return self.payload.get(key)

    @utility.deprecated
    def todense(self) -> VectorizedCorpus:
        """Returns dense BoW matrix DO NOT USE IF LARGE CORPUS!"""
        dtm = self._bag_term_matrix

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
        return self._bag_term_matrix[:, self.token2id[word]].todense().A1  # x.A1 == np.asarray(x).ravel()

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

        document_index = self.document_index[self.document_index.apply(px, axis=1)]

        indices = list(document_index.index)

        corpus = VectorizedCorpus(
            bag_term_matrix=self._bag_term_matrix[indices, :],
            token2id=self.token2id,
            document_index=document_index,
            **self.payload,
        )

        return corpus

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
        btm = sklearn.preprocessing.normalize(self._bag_term_matrix, axis=axis, norm=norm)

        if keep_magnitude is True:
            factor = self._bag_term_matrix[0, :].sum() / btm[0, :].sum()
            btm = btm * factor

        corpus = VectorizedCorpus(
            bag_term_matrix=btm,
            token2id=self.token2id,
            document_index=self.document_index,
            overridden_term_frequency=self._overridden_term_frequency,
            **self.payload,
        )

        return corpus

    def normalize_by_raw_counts(self):

        if 'n_raw_tokens' not in self.document_index.columns:
            # logging.warning("Normalizing using DTM counts (not actual self counts)")
            # return self.normalize()
            raise VectorizedCorpusError("raw count normalize attempted but no n_raw_tokens in document index")

        token_counts = self.document_index.n_raw_tokens.values
        btm = utility.normalize_sparse_matrix_by_vector(self._bag_term_matrix, token_counts)
        corpus = VectorizedCorpus(
            bag_term_matrix=btm,
            token2id=self.token2id,
            document_index=self.document_index,
            overridden_term_frequency=self._overridden_term_frequency,
            **self.payload,
        )

        return corpus

    def token_indices(self, tokens: Iterable[str]) -> List[int]:
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
        return [self.token2id[token] for token in tokens if token in self.token2id]

    def tf_idf(self, norm: str = 'l2', use_idf: bool = True, smooth_idf: bool = True) -> IVectorizedCorpus:
        """Returns a (normalized) TF-IDF transformed version of the corpus

        Calls sklearn's TfidfTransformer
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn-feature-extraction-text-tfidftransformer
        https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
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

        tfidf_bag_term_matrix = transformer.fit_transform(self._bag_term_matrix)

        n_corpus = VectorizedCorpus(
            bag_term_matrix=tfidf_bag_term_matrix,
            token2id=self.token2id,
            document_index=self.document_index,
            overridden_term_frequency=self._overridden_term_frequency,
            **self.payload,
        )

        return n_corpus

    def to_bag_of_terms(self, indices: Optional[Iterable[int]] = None) -> Iterable[Iterable[str]]:
        """Returns a document token stream that corresponds to the BoW.
        Tokens are repeated according to BoW token counts.
        Note: Will not work on a normalized corpus!

        Parameters
        ----------
        indices : Optional[Iterable[int]], optional
            Specifies word subset, by default None

        Returns
        -------
        Iterable[Iterable[str]]
            Documenttoken stream.
        """
        dtm = self._bag_term_matrix
        indices = indices or range(0, dtm.shape[0])
        id2token = self.id2token
        return (
            (w for ws in (dtm[doc_id, i] * [id2token[i]] for i in dtm[doc_id, :].nonzero()[1]) for w in ws)
            for doc_id in indices
        )

    def co_occurrence_matrix(self) -> scipy.sparse.spmatrix:
        """Computes (document) cooccurence matrix

        Returns
        -------
        Tuple[scipy.sparce.spmatrix. Dict[int,str]]
            The co-occurrence matrix
        """
        term_term_matrix = np.dot(self._bag_term_matrix.T, self._bag_term_matrix)
        term_term_matrix = scipy.sparse.triu(term_term_matrix, 1)

        return term_term_matrix

    def find_matching_words(self, word_or_regexp: Set[str], n_max_count: int, descending: bool = False) -> List[str]:
        """Returns words in corpus that matches candidate tokens """
        words = self.pick_n_top_words(
            find_matching_words_in_vocabulary(self.token2id, word_or_regexp),
            n_max_count,
            descending=descending,
        )
        return words

    def find_matching_words_indices(
        self, word_or_regexp: List[str], n_max_count: int, descending: bool = False
    ) -> List[int]:
        """Returns `tokens´ indices` in corpus that matches candidate tokens """
        indices: List[int] = [
            self.token2id[token]
            for token in self.find_matching_words(word_or_regexp, n_max_count, descending=descending)
            if token in self.token2id
        ]
        return indices

    @staticmethod
    def create(
        bag_term_matrix: scipy.sparse.csr_matrix,
        token2id: Dict[str, int],
        document_index: DocumentIndex,
        overridden_term_frequency: Dict[str, int] = None,
        **kwargs,
    ) -> "IVectorizedCorpus":
        return VectorizedCorpus(
            bag_term_matrix=bag_term_matrix,
            token2id=token2id,
            document_index=document_index,
            overridden_term_frequency=overridden_term_frequency,
            **kwargs,
        )

    def nbytes(self, kind="bytes") -> float:
        k = {'bytes': 0, 'kb': 1, 'mb': 2, 'gb': 3}.get(kind.lower(), 0)
        with contextlib.suppress(Exception):
            return (self.data.data.nbytes + self.data.indptr.nbytes + self.data.indices.nbytes) / pow(1024, k)
        return None

    def zero_out_by_tf_threshold(self, tf_threshold: Union[int, float]) -> Sequence[int]:
        """Clears (inplace) tokens (columns) having a TF-value (column sum) less than threshold
        Does not change shape."""
        indices = np.argwhere(self.term_frequency < tf_threshold).ravel()
        if len(indices) > 0:
            self.zero_out_by_indices(indices)
        return indices

    def zero_out_by_others_zeros(self, other: VectorizedCorpus) -> VectorizedCorpus:
        """Zeroes out elements in `self` where corresponding element in `other` is zero
        Doe's not change shape."""
        mask = other.data > 0
        data = self.data
        data = data.multiply(mask)
        data.eliminate_zeros()
        return self

    def zero_out_by_indices(self, indices: Sequence[int]) -> Sequence[int]:
        """Zeros out values for given column indicies"""

        if indices is None or len(indices) == 0:
            return indices

        term_frequency = self.term_frequency

        indices: Sequence[int] = [i for i in indices if term_frequency[i] > 0]

        if len(indices) == 0:
            return indices

        data = self._bag_term_matrix.tolil()
        data[:, indices] = 0
        data = data.tocsr()
        data.eliminate_zeros()

        self._bag_term_matrix = data
        return indices

    # def zero_out_by_indices(self, indices: Sequence[int]) -> None:
    #     """Zeros out columns for indicies"""
    #     if indices is None or len(indices) == 0:
    #         return

    #     self.data[:, indices] = 0
    #     self.data.eliminate_zeros()


def find_matching_words_in_vocabulary(token2id: Mapping[str], candidate_words: Set[str]) -> Set[str]:

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
