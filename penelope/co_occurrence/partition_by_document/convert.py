from typing import Set

import numpy as np
import pandas as pd
import scipy
from penelope.corpus import DocumentIndex, Token2Id, VectorizedCorpus
from penelope.type_alias import CoOccurrenceDataFrame

# from ..interface import PartitionKeyNotUniqueKey

# def compute_normalized_count(co_occurrences: pd.DataFrame, document_index: DocumentIndex) -> pd.Series:

#     if document_index is None:
#         return co_occurrences

#     document_index = document_index.set_index('document_id', drop=False)

#     # FIXME Add year and value_n_t/value_n_r_t to co
#     co_occurrences['year'] = co_occurrences.merge()
#     co_occurrences['value_n_t'] = co_occurrences['value'] / co_occurrences.groupby('year')['value'].transform('sum')

#     co_occurrences.merge(document_index, )

#     for n_token_count, target_field in [('n_tokens', 'value_n_t'), ('n_raw_tokens', 'value_n_r_t')]:
#         if n_token_count in document_index.columns:
#             try:
#                 co_occurrences[target_field] = co_occurrences.value / float(sum(document_index[n_token_count].values))
#             except ZeroDivisionError:
#                 co_occurrences[target_field] = 0.0
#         else:
#             logger.warning(f"{target_field}: cannot compute since {n_token_count} not in corpus document index")


def truncate_by_global_threshold(co_occurrences: pd.DataFrame, threshold: int) -> pd.DataFrame:
    if len(co_occurrences) == 0:
        return co_occurrences
    if threshold is None or threshold <= 1:
        return co_occurrences
    filtered_co_occurrences = co_occurrences[
        co_occurrences.groupby(["w1_id", "w2_id"])['value'].transform('sum') >= threshold
    ]
    return filtered_co_occurrences


def co_occurrence_term_term_matrix_to_dataframe(
    term_term_matrix: scipy.sparse.spmatrix,
    threshold_count: int = 1,
    ignore_ids: Set[int] = None,
    dtype: np.dtype = np.uint32,
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

    if co_occurrences.value.max() < np.iinfo(np.uint16).max:
        co_occurrences['value'] = co_occurrences.value.astype(np.uint16)

    if ignore_ids:
        co_occurrences = co_occurrences[
            (~co_occurrences.w1_id.isin(ignore_ids) & ~co_occurrences.w2_id.isin(ignore_ids))
        ]

    if threshold_count > 1:
        co_occurrences = co_occurrences[co_occurrences.value >= threshold_count]

    return co_occurrences


def co_occurrence_dataframe_to_vectorized_corpus(
    *,
    co_occurrences: CoOccurrenceDataFrame,
    document_index: DocumentIndex,
    token2id: Token2Id,
) -> VectorizedCorpus:
    """Creates a DTM corpus from a co-occurrence result set that was partitioned by `partition_column`."""

    """Create distinct word-pair tokens and assign a token_id"""
    to_token = token2id.id2token.get
    token_pairs: pd.DataFrame = co_occurrences[["w1_id", "w2_id"]].drop_duplicates().reset_index(drop=True)
    token_pairs["token_id"] = token_pairs.index
    token_pairs["token"] = token_pairs.w1_id.apply(to_token) + "/" + token_pairs.w2_id.apply(to_token)

    """Create a new vocabulary"""
    vocabulary = token_pairs.set_index("token").token_id.to_dict()

    """Merge and assign token_id to co-occurring pairs"""
    token_ids: pd.Series = co_occurrences.merge(
        token_pairs.set_index(['w1_id', 'w2_id']),
        how='left',
        left_on=['w1_id', 'w2_id'],
        right_index=True,
    ).token_id

    """Set document_id as unique key for DTM document index """
    document_index = document_index.set_index('document_id', drop=False).rename_axis('').sort_index()

    """Make certain that the matrix gets right shape (to avoid offset errors)"""
    shape = (len(document_index), len(vocabulary))
    matrix = scipy.sparse.coo_matrix(
        (
            co_occurrences.value.astype(np.uint16),
            (
                co_occurrences.document_id.astype(np.uint32),
                token_ids.astype(np.uint32),
            ),
        ),
        shape=shape,
    )

    """Create the final corpus"""
    corpus = VectorizedCorpus(matrix, token2id=vocabulary, document_index=document_index)

    return corpus
