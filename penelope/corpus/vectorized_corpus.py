from __future__ import annotations

import logging
import os
import pickle
import time
from heapq import nlargest
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn.preprocessing
import textacy
from penelope.utility import read_json, write_json
from sklearn.feature_extraction.text import TfidfTransformer

# pylint: disable=logging-format-interpolation, too-many-public-methods

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("")


class VectorizedCorpus:
    def __init__(
        self,
        bag_term_matrix: scipy.sparse.csr_matrix,
        token2id: Dict[str, int],
        documents: pd.DataFrame,
        word_counts: Dict[str, int] = None,
    ):
        """Class that encapsulates a bag-of-word matrix.

        Parameters
        ----------
        bag_term_matrix : scipy.sparse.csr_matrix
            The bag-of-word matrix
        token2id : dict(str, int)
            Token to token id translation i.e. translates token to column index
        documents : pd.DataFrame
            Corpus document metadata (bag-of-word row metadata)
        word_counts : dict(str,int), optional
            Total corpus word counts, by default None, computed if None
        """

        # Ensure that we have a sparse matrix (CSR)
        if not scipy.sparse.issparse(bag_term_matrix):
            bag_term_matrix = scipy.sparse.csr_matrix(bag_term_matrix)
        elif not scipy.sparse.isspmatrix_csr(bag_term_matrix):
            bag_term_matrix = bag_term_matrix.tocsr()

        self.bag_term_matrix: scipy.sparse.csr_matrix = bag_term_matrix

        assert scipy.sparse.issparse(self.bag_term_matrix), "only sparse data allowed"

        self.token2id = token2id
        self.id2token_ = None
        self._documents = documents
        self.word_counts = word_counts

        if self.word_counts is None:

            # Compute word counts
            Xsum = self.bag_term_matrix.sum(axis=0)
            Xsum = np.ravel(Xsum)

            self.word_counts = {w: Xsum[i] for w, i in self.token2id.items()}
            # self.id2token = { i: t for t,i in self.token2id.items()}

        # n_bags = self.bag_term_matrix.shape[0]
        # n_vocabulary = self.bag_term_matrix.shape[1]
        # n_tokens = sum(self.word_counts.values())
        # logger.info('#bags: {}, #vocab: {}, #tokens: {}'.format(n_bags, n_vocabulary, n_tokens))

    @property
    def id2token(self) -> Mapping[int, str]:
        if self.id2token_ is None and self.token2id is not None:
            self.id2token_ = {i: t for t, i in self.token2id.items()}
        return self.id2token_

    @property
    def vocabulary(self):
        vocab = [self.id2token[i] for i in range(0, self.data.shape[1])]
        return vocab

    @property
    def T(self) -> scipy.sparse.csr_matrix:
        """Returns transpose of BoW matrix """
        return self.bag_term_matrix.T

    @property
    def data(self) -> scipy.sparse.csr_matrix:
        """Returns BoW matrix """
        return self.bag_term_matrix

    @property
    def term_bag_matrix(self) -> scipy.sparse.csr_matrix:
        """Returns transpose of BoW matrix """
        return self.bag_term_matrix.T

    @property
    def n_docs(self) -> int:
        """Returns number of documents """
        return self.bag_term_matrix.shape[1]

    @property
    def n_terms(self) -> int:
        """Returns number of types (unique words) """
        return self.bag_term_matrix.shape[0]

    @property
    def documents(self) -> pd.DataFrame:
        """Returns number document index (part of interface) """
        return self._documents

    def todense(self) -> VectorizedCorpus:
        """Returns dense BoW matrix"""
        dtm = self.data

        if scipy.sparse.issparse(dtm):
            dtm = dtm.todense()

        if isinstance(dtm, np.matrix):
            dtm = np.asarray(dtm)

        self.bag_term_matrix = dtm

        return self

    def dump(self, *, tag: str, folder: str, compressed: bool = True) -> VectorizedCorpus:
        """Store corpus on disk.

        The file is stored as two files: one that contains the BoW matrix (.npy or .npz)
        and a pickled file that contains dictionary, word counts and the document index

        The two files are stored in files with names based on the specified `tag`:

            {tag}_vectorizer_data.pickle         Metadata `token2id`, `documents` and `word_counts`
            {tag}_vector_data.[npz|npy]          The document-term matrix (numpy or sparse format)


        Parameters
        ----------
        tag : str, optional
            String to be prepended to file name, set to timestamp if None
        folder : str, optional
            Target folder, by default './output'
        compressed : bool, optional
            Specifies if matrix is store as .npz or .npy, by default .npz

        """
        tag = tag or time.strftime("%Y%m%d_%H%M%S")

        data = {
            'token2id': self.token2id,
            'word_counts': self.word_counts,
            'documents': self.documents,
        }
        data_filename = VectorizedCorpus._data_filename(tag, folder)

        with open(data_filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        matrix_filename = VectorizedCorpus._matrix_filename(tag, folder)

        if compressed:
            assert scipy.sparse.issparse(self.bag_term_matrix)
            scipy.sparse.save_npz(matrix_filename, self.bag_term_matrix, compressed=True)
        else:
            np.save(matrix_filename + '.npy', self.bag_term_matrix, allow_pickle=True)

        return self

    # FIXME: #15 Make arguments naming mandatory to avoid mixups
    @staticmethod
    def dump_exists(*, tag: str, folder: str) -> bool:
        """Checks if corpus with tag `tag` exists in folder `folder`

        Parameters
        ----------
        tag : str
            Corpus prefix tag
        folder : str, optional
            Corpus folder to look in, by default './output'
        """
        return os.path.isfile(VectorizedCorpus._data_filename(tag, folder))

    @staticmethod
    def remove(*, tag: str, folder: str):
        Path(os.path.join(folder, f'{tag}_vector_data.npz')).unlink(missing_ok=True)
        Path(os.path.join(folder, f'{tag}_vector_data.npy')).unlink(missing_ok=True)
        Path(os.path.join(folder, f'{tag}_vectorizer_data.pickle')).unlink(missing_ok=True)
        Path(os.path.join(folder, f'{tag}_vectorizer_data.json')).unlink(missing_ok=True)

    @staticmethod
    def load(*, tag: str, folder: str) -> VectorizedCorpus:
        """Loads corpus with tag `tag` in folder `folder`

        Raises `FileNotFoundError` if any of the two files containing metadata and matrix doesn't exist.

        Two files are loaded based on specified `tag`:

            {tag}_vectorizer_data.pickle         Contains metadata `token2id`, `documents` and `word_counts`
            {tag}_vector_data.[npz|npy]          Contains the document-term matrix (numpy or sparse format)


        Parameters
        ----------
        tag : str
            Corpus prefix tag
        folder : str, optional
            Corpus folder to look in, by default './output'

        Returns
        -------
        VectorizedCorpus
            Loaded corpus
        """
        data_filename = VectorizedCorpus._data_filename(tag, folder)
        with open(data_filename, 'rb') as f:
            data = pickle.load(f)

        token2id: Mapping = data["token2id"]
        documents: pd.DataFrame = data.get("documents", data.get("document_index", None))

        matrix_basename = VectorizedCorpus._matrix_filename(tag, folder)

        if os.path.isfile(matrix_basename + '.npz'):
            bag_term_matrix = scipy.sparse.load_npz(matrix_basename + '.npz')
        else:
            bag_term_matrix = np.load(matrix_basename + '.npy', allow_pickle=True).item()

        return VectorizedCorpus(bag_term_matrix, token2id=token2id, documents=documents)

    @staticmethod
    def dump_options(*, tag: str, folder: str, options: Dict):
        json_filename = os.path.join(folder, f"{tag}_vectorizer_data.json")
        write_json(json_filename, options)

    @staticmethod
    def load_options(*, tag: str, folder: str) -> Dict:
        """Loads vectrize options if they exists"""
        json_filename = os.path.join(folder, f"{tag}_vectorizer_data.json")
        if os.path.isfile(json_filename):
            return read_json(json_filename)
        return dict()

    @staticmethod
    def _data_filename(tag: str, folder: str) -> str:
        """Returns pickled basename for given tag and folder"""
        return os.path.join(folder, f"{tag}_vectorizer_data.pickle")

    @staticmethod
    def _matrix_filename(tag: str, folder: str) -> str:
        """Returns BoW matrix basename for given tag and folder"""
        return os.path.join(folder, f"{tag}_vector_data")

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

    def collapse_by_category(
        self, column, X=None, df: pd.DataFrame = None, aggregate_function: str = 'sum', dtype=np.float
    ):  # -> VectorizedCorpus:
        """Sums ups all rows in based on each row's index having same value in column `column`in data frame `df`

        Parameters
        ----------
        column : str
            The categorical column in `df` to be used in the grouping of rows in `X`

        X : np.ndarray(N, M), optional
            Matrix of shape (N, M), by default None

        df : DataFrame, optional
            DataFrame of size N, where each row `ì` contains data that describes row `i` in `X`, by default None

        aggregate_function : str, optional, values `sum` or `mean`
            DataFrame of size N, where each row `ì` contains data that describes row `i` in `X`, by default None

        Returns
        -------
        tuple: np.ndarray(K, M), List[Any]
            A matrix of size K wherw K is the number of unique categorical values in `df[column]`
            A list of length K of category values, where i:th value is category of i:th row in returned matrix
        """

        X = self.bag_term_matrix if X is None else X
        df = self.documents if df is None else df

        assert aggregate_function in {'sum', 'mean'}
        assert X.shape[0] == len(df)

        categories = list(sorted(df[column].unique().tolist()))

        Y = np.zeros((len(categories), X.shape[1]), dtype=(dtype or X.dtype))

        for i, value in enumerate(categories):

            indices = list((df.loc[df[column] == value].index))

            if aggregate_function == 'mean':
                Y[i, :] = X[indices, :].mean(axis=0)
            else:
                Y[i, :] = X[indices, :].sum(axis=0)

        return Y, categories

    # @jit
    # CONSIDER: Refactor away function (make use of `collapse_by_category`)
    def group_by_year(self) -> VectorizedCorpus:
        """Returns a new corpus where documents have been grouped and summed up by year."""

        X = self.bag_term_matrix  # if X is None else X
        df = self.documents  # if df is None else df

        min_value, max_value = df.year.min(), df.year.max()

        Y = np.zeros(((max_value - min_value) + 1, X.shape[1]))

        for i in range(0, Y.shape[0]):  # pylint: disable=unsubscriptable-object

            indices = list((df.loc[df.year == min_value + i].index))

            if len(indices) > 0:
                Y[i, :] = X[indices, :].sum(axis=0)

        years = list(range(min_value, max_value + 1))
        documents = pd.DataFrame({'year': years, 'filename': map(str, years)})

        v_corpus = VectorizedCorpus(Y, token2id=self.token2id, documents=documents, word_counts=self.word_counts)

        return v_corpus

    # CONSIDER: Refactor away function (make use of `collapse_by_category`)
    def group_by_year2(self, aggregate_function='sum', dtype=None) -> VectorizedCorpus:
        """Variant of `group_by_year` where aggregate function can be specified."""

        assert aggregate_function in {'sum', 'mean'}

        X = self.bag_term_matrix  # if X is None else X
        df = self.documents  # if df is None else df

        min_value, max_value = df.year.min(), df.year.max()

        Y = np.zeros(((max_value - min_value) + 1, X.shape[1]), dtype=(dtype or X.dtype))

        for i in range(0, Y.shape[0]):  # pylint: disable=unsubscriptable-object

            indices = list((df.loc[df.year == min_value + i].index))

            if len(indices) > 0:
                if aggregate_function == 'mean':
                    Y[i, :] = X[indices, :].mean(axis=0)
                else:
                    Y[i, :] = X[indices, :].sum(axis=0)

                # Y[i,:] = self._group_aggregate_functions[aggregate_function](X[indices,:], axis=0)

        years = list(range(min_value, max_value + 1))

        documents = pd.DataFrame({'year': years, 'filename': map(str, years)})

        v_corpus = VectorizedCorpus(Y, self.token2id, documents, self.word_counts)

        return v_corpus

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

        meta_documents = self.documents[self.documents.apply(px, axis=1)]

        indices = list(meta_documents.index)

        v_corpus = VectorizedCorpus(self.bag_term_matrix[indices, :], self.token2id, meta_documents, None)

        return v_corpus

    # @jit
    def normalize(self, axis: int = 1, norm: str = 'l1', keep_magnitude: bool = False) -> VectorizedCorpus:
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
        normalized_bag_term_matrix = sklearn.preprocessing.normalize(self.bag_term_matrix, axis=axis, norm=norm)

        if keep_magnitude is True:
            factor = self.bag_term_matrix[0, :].sum() / normalized_bag_term_matrix[0, :].sum()
            normalized_bag_term_matrix = normalized_bag_term_matrix * factor

        v_corpus = VectorizedCorpus(normalized_bag_term_matrix, self.token2id, self.documents, self.word_counts)

        return v_corpus

    def n_top_tokens(self, n_top) -> Dict[str, int]:
        """Returns `n_top` most frequent words.

        Parameters
        ----------
        n_top : int
            Number of words to return

        Returns
        -------
        Dict[str, int]
            Most frequent words and their counts, subset of dict `word_counts`

        """
        tokens = {w: self.word_counts[w] for w in nlargest(n_top, self.word_counts, key=self.word_counts.get)}
        return tokens

    # @autojit
    def slice_by_n_count(self, n_count: int) -> VectorizedCorpus:
        """Create a subset corpus where words having a count less than 'n_count' are removed

        Parameters
        ----------
        n_count : int
            Specifies min word count to keep.

        Returns
        -------
        VectorizedCorpus
            Subset of self where words having a count less than 'n_count' are removed
        """

        tokens = set(w for w, c in self.word_counts.items() if c >= n_count)

        def _px(w):
            return w in tokens

        return self.slice_by(_px)

    def slice_by_n_top(self, n_top) -> VectorizedCorpus:
        """Create a subset corpus that only contains most frequent `n_top` words

        Parameters
        ----------
        n_top : int
            Specifies specifies number of top words to keep.

        Returns
        -------
        VectorizedCorpus
            Subset of self where words having a count less than 'n_count' are removed
        """
        tokens = set(nlargest(n_top, self.word_counts, key=self.word_counts.get))

        def _px(w):
            return w in tokens

        return self.slice_by(_px)

    # def doc_freqs(self):
    #     """ Count number of occurrences of each value in array of non-negative ints. """
    #     return np.bincount(self.doc_term_matrix.indices, minlength=self.n_terms)

    # def slice_by_document_frequency(self, min_df, max_df):
    #     min_doc_count = min_df if isinstance(min_df, int) else int(min_df * self.n_docs)
    #     dfs = self.doc_freqs()
    #     mask = np.ones(self.n_terms, dtype=bool)
    #     if min_doc_count > 1:
    #         mask &= dfs >= min_doc_count
    #     # map old term indices to new ones
    #     new_indices = np.cumsum(mask) - 1
    #     token2id = {
    #         term: new_indices[old_index]
    #         for term, old_index in self.token2id.items()
    #             if mask[old_index]
    #     }
    #     kept_indices = np.where(mask)[0]
    #     return (self.bag_term_matrix[:, kept_indices], token2id)

    def slice_by_document_frequency(self, max_df=1.0, min_df=1, max_n_terms=None) -> VectorizedCorpus:
        """Creates a subset corpus where common/rare terms are filtered out.

        Textacy util function filter_terms_by_df is used for the filtering.

        See https://chartbeat-labs.github.io/textacy/build/html/api_reference/vsm_and_tm.html.

        Parameters
        ----------
        max_df : float, optional
            Max number of docs or fraction of total number of docs, by default 1.0
        min_df : int, optional
            Max number of docs or fraction of total number of docs, by default 1
        max_n_terms : in optional
            [description], by default None
        """
        sliced_bag_term_matrix, token2id = textacy.vsm.matrix_utils.filter_terms_by_df(
            self.bag_term_matrix, self.token2id, max_df=max_df, min_df=min_df, max_n_terms=max_n_terms
        )
        word_counts = {w: c for w, c in self.word_counts.items() if w in token2id}

        v_corpus = VectorizedCorpus(sliced_bag_term_matrix, token2id, self.documents, word_counts)

        return v_corpus

    # @autojit
    def slice_by(self, px) -> VectorizedCorpus:
        """Create a subset corpus based on predicate `px`

        Parameters
        ----------
        px : str -> bool
            Predicate that tests if a word should be kept.

        Returns
        -------
        VectorizedCorpus
            Subset containing words for which `px` evaluates to true.
        """
        indices = [self.token2id[w] for w in self.token2id.keys() if px(w)]

        indices.sort()

        sliced_bag_term_matrix = self.bag_term_matrix[:, indices]
        token2id = {self.id2token[indices[i]]: i for i in range(0, len(indices))}
        word_counts = {w: c for w, c in self.word_counts.items() if w in token2id}

        v_corpus = VectorizedCorpus(sliced_bag_term_matrix, token2id, self.documents, word_counts)

        return v_corpus

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

    def to_n_top_dataframe(self, n_top: int):
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
        columns = list(v_n_corpus.bag_term_matrix.index)
        df = pd.DataFrame(data=data, index=[v_n_corpus.id2token[i] for i in range(0, n_top)], columns=columns)
        return df

    def year_range(self) -> Tuple[Optional[int], Optional[int]]:
        """Returns document's year range

        Returns
        -------
        Tuple[Optional[int],Optional[int]]
            Min/max document year
        """
        if 'year' in self.documents.columns:
            return (self.documents.year.min(), self.documents.year.max())
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

    def tf_idf(self, norm: str = 'l2', use_idf: bool = True, smooth_idf: bool = True) -> VectorizedCorpus:
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

        n_corpus = VectorizedCorpus(tfidf_bag_term_matrix, self.token2id, self.documents, self.word_counts)

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
        """Returns a document token stream that

        Parameters
        ----------
        indicies : Iterable[int], optional
            [description], by default None

        Returns
        -------
        Iterable[Iterable[str]]
            [description]
        """
        """
        List the top n words in a subset of the corpus sorted according to occurrence.

        """
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


def load_corpus(
    *,
    tag: str,
    folder: str,
    n_count: int = 10000,
    n_top: int = 100000,
    axis: Optional[int] = 1,
    keep_magnitude: bool = True,
) -> VectorizedCorpus:
    """Loads a previously saved vectorized corpus from disk. Easaly the best loader ever.

    Parameters
    ----------
    tag : str
        Corpus filename prefix
    folder : str
        Source folder where corpus reside
    n_count : int, optional
        Words having a (global) count below this limit are discarded, by default 10000
    n_top : int, optional
        Only the 'n_top' words sorted by word counts should be loaded, by default 100000
    axis : int, optional
        Axis used to normalize the data along. 1 normalizes each row (bag/document), 0 normalizes each column (word).
    keep_magnitude : bool, optional
        Scales result matrix so that sum equals input matrix sum, by default True

    Returns
    -------
    VectorizedCorpus
        The loaded corpus
    """
    v_corpus = VectorizedCorpus.load(tag=tag, folder=folder).group_by_year()

    if n_count is not None:
        v_corpus = v_corpus.slice_by_n_count(n_count)

    if n_top is not None:
        v_corpus = v_corpus.slice_by_n_top(n_top)

    if axis is not None and v_corpus.data.shape[1] > 0:
        v_corpus = v_corpus.normalize(axis=axis, keep_magnitude=keep_magnitude)

    return v_corpus


def load_cached_normalized_vectorized_corpus(tag, folder, n_count=10000, n_top=100000, keep_magnitude=True):

    year_cache_tag = "cached_year_{}_{}".format(tag, "km" if keep_magnitude else "")

    v_corpus = None

    if not VectorizedCorpus.dump_exists(tag=year_cache_tag, folder=folder):
        logger.info("Caching corpus grouped by year...")
        v_corpus = (
            VectorizedCorpus.load(tag=tag, folder=folder)
            .group_by_year()
            .normalize(axis=1, keep_magnitude=keep_magnitude)
            .dump(tag=year_cache_tag, folder=folder)
        )

    if v_corpus is None:
        v_corpus = (
            VectorizedCorpus.load(tag=year_cache_tag, folder=folder).slice_by_n_count(n_count).slice_by_n_top(n_top)
        )

    return v_corpus
