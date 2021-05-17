import numpy as np
import pandas as pd
import scipy


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

    if threshold_count > 1:
        co_occurrences = co_occurrences[co_occurrences.value >= threshold_count]

    return co_occurrences
