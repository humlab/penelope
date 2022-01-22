from typing import Mapping, Set, Union

import numpy as np
import pandas as pd
import scipy

from penelope.corpus import CorpusVectorizer, ITokenizedCorpus, Token2Id, TokenizedCorpus, VectorizedCorpus
from penelope.corpus.readers import ICorpusReader
from penelope.type_alias import CoOccurrenceDataFrame


def to_co_occurrence_matrix(
    corpus_or_reader: Union[ICorpusReader, TokenizedCorpus], vocabulary: Mapping[str, int] = None
) -> scipy.sparse.spmatrix:
    """Computes a term-term co-ocurrence matrix for documents in corpus/reader.

    Parameters
    ----------
    corpus_or_reader : Union[ICorpusReader,TokenizedCorpus]
        Sequence of tokenized documents

    Returns
    -------
    pd.DataFrame
        Upper diagonal of term-term frequency matrix (TTM). Note that diagonal (wi, wi) is not returned
    """

    if not isinstance(corpus_or_reader, ITokenizedCorpus):
        corpus_or_reader = TokenizedCorpus(reader=corpus_or_reader)

    vocabulary = vocabulary or corpus_or_reader.token2id
    dtm_corpus: VectorizedCorpus = CorpusVectorizer().fit_transform(
        corpus_or_reader, already_tokenized=True, vocabulary=vocabulary
    )
    term_term_matrix = dtm_corpus.co_occurrence_matrix()
    return term_term_matrix


def truncate_by_global_threshold(co_occurrences: pd.DataFrame, threshold: int) -> pd.DataFrame:
    if len(co_occurrences) == 0:
        return co_occurrences
    if threshold is None or threshold <= 1:
        return co_occurrences
    filtered_co_occurrences = co_occurrences[
        co_occurrences.groupby(["w1_id", "w2_id"])['value'].transform('sum') >= threshold
    ]
    return filtered_co_occurrences


def term_term_matrix_to_co_occurrences(
    term_term_matrix: scipy.sparse.spmatrix,
    threshold_count: int = 1,
    ignore_ids: Set[int] = None,
    dtype: np.dtype = np.int32,
) -> pd.DataFrame:
    """Converts a TTM to a Pandas DataFrame

    Args:
        term_term_matrix (scipy.sparse.spmatrix): [description]
        threshold_count (int, optional): min threshold for global token count. Defaults to 1.

    Returns:
        pd.DataFrame: co-occurrence data frame
    """

    co_occurrences = (
        pd.DataFrame(
            {
                'w1_id': pd.Series(term_term_matrix.row, dtype=dtype),
                'w2_id': pd.Series(term_term_matrix.col, dtype=dtype),
                'value': term_term_matrix.data,
            },
            dtype=dtype,
        )
        .sort_values(['w1_id', 'w2_id'])
        .reset_index(drop=True)
    )

    if co_occurrences.value.max() < np.iinfo(np.int32).max:
        co_occurrences['value'] = co_occurrences.value.astype(np.int32)

    if ignore_ids:
        co_occurrences = co_occurrences[
            (~co_occurrences.w1_id.isin(ignore_ids) & ~co_occurrences.w2_id.isin(ignore_ids))
        ]

    if threshold_count > 1:
        co_occurrences = co_occurrences[co_occurrences.value >= threshold_count]

    return co_occurrences


def co_occurrence_corpus_to_co_occurrence(
    *,
    coo_corpus: VectorizedCorpus,
    token2id: Token2Id,
) -> CoOccurrenceDataFrame:
    """Creates a co-occurrence data frame from a co-occurrence DTM corpus."""
    return coo_corpus.to_co_occurrences(token2id)


def to_token_window_counts_matrix(counters: Mapping[int, Mapping[int, int]], shape: tuple) -> scipy.sparse.spmatrix:
    """Create a matrix with token's window count for each document (rows).
       The shape of the returned sparse matrix is [number of document, vocabulary size]

    Args:
        counters (dict): Dict (key document id) of dict (key token id) of window counts
        shape (tuple): Size of returned sparse matrix

    Returns:
        scipy.sparse.spmatrix: window counts matrix
    """

    matrix: scipy.sparse.lil_matrix = scipy.sparse.lil_matrix(shape, dtype=np.uint16)

    for document_id, counts in counters.items():
        matrix[document_id, list(counts.keys())] = list(counts.values())

    return matrix.tocsr()
