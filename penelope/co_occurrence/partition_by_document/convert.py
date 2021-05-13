import numpy as np
import pandas as pd
import scipy


def term_term_matrix_to_dataframe(
    term_term_matrix: scipy.sparse.spmatrix,
    threshold_count: int = 1,
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
            {'w1_id': term_term_matrix.row, 'w2_id': term_term_matrix.col, 'value': term_term_matrix.data},
            dtype=np.uint32,
        )[['w1_id', 'w2_id', 'value']]
        .sort_values(['w1_id', 'w2_id'])
        .reset_index(drop=True)
    )

    if threshold_count > 1:
        co_occurrences = co_occurrences[co_occurrences.value >= threshold_count]

    return co_occurrences


# def compute_value_n_t(co_occurrences: pd.DataFrame, document_index: DocumentIndex):
#     if document_index is None:
#         return co_occurrences
#     for n_token_count, target_field in [('n_tokens', 'value_n_t'), ('n_raw_tokens', 'value_n_r_t')]:
#         if n_token_count in document_index.columns:
#             try:
#                 co_occurrences[target_field] = co_occurrences.value / float(sum(document_index[n_token_count].values))
#             except ZeroDivisionError:
#                 co_occurrences[target_field] = 0.0
#         else:
#             logger.warning(f"{target_field}: cannot compute since {n_token_count} not in corpus document catalogue")
