import numpy as np
import pandas as pd
import scipy

# @deprecated
# def compute_hal_score_cellwise(nw_xy: scipy.sparse.spmatrix, nw_x: scipy.sparse.spmatrix, vocab_mapping: dict) -> scipy.sparse.spmatrix:
#     # Cell by CELL
#     for d in range(0, nw_xy.shape[0]):
#         for t in range(0, nw_xy.shape[1]):
#             w1_id = vocab_mapping[t][0]
#             w2_id = vocab_mapping[t][1]
#             nw_xy[d, t] = nw_xy[d, t] / (nw_x[d, w1_id] + nw_x[d, w2_id] - nw_xy[d, t])

#     return nw_xy

# @deprecated
# def compute_hal_score_colwise(nw_xy: scipy.sparse.spmatrix, nw_x: scipy.sparse.spmatrix, vocab_mapping: dict) -> scipy.sparse.spmatrix:
#     # COL by COL
#     for t in range(0, nw_xy.shape[1]):
#         nw_xy[:, t] = nw_xy[:, t]  / (nw_x[:,vocab_mapping[t][0]] + nw_x[:,vocab_mapping[t][1]] - nw_xy[:, t])
#     return nw_xy


def compute_hal_score_by_co_occurrence_matrix(
    co_occurrences: pd.DataFrame, nw_x_matrix: scipy.sparse.spmatrix
) -> pd.Series:
    """Compute yearly HAL-score for each co-occurrence pair (w1, w2)"""

    # nw_xy is given by co_occurrences data frame
    co_occurrences = co_occurrences[['document_id', 'w1_id', 'w2_id', 'value']]
    co_occurrences['nw_xy'] = co_occurrences.value

    nw_x_matrix: scipy.sparse.spmatrix = nw_x_matrix.tocoo()
    nw_x_frame: pd.DataFrame = pd.DataFrame(
        data={
            'document_id': nw_x_matrix.row,
            'token_id': nw_x_matrix.col,
            'nw_x': nw_x_matrix.data,
        },
    ).set_index(['document_id', 'token_id'])

    co_occurrences['nw_x'] = co_occurrences[['document_id', 'w1_id']].merge(
        nw_x_frame, how='left', left_on=['document_id', 'w1_id'], right_index=True
    )['nw_x']
    co_occurrences['nw_y'] = co_occurrences[['document_id', 'w2_id']].merge(
        nw_x_frame, how='left', left_on=['document_id', 'w2_id'], right_index=True
    )['nw_x']

    cwr: pd.Series = co_occurrences.nw_xy / (co_occurrences.nw_x + co_occurrences.nw_y - co_occurrences.nw_xy)

    cwr.fillna(0.0, inplace=True)
    cwr.clip(lower=0.0, upper=None, axis=None, inplace=True)

    return cwr


def compute_hal_cwr_score(
    nw_xy: scipy.sparse.spmatrix,
    nw_x: scipy.sparse.spmatrix,
    vocab_mapping: dict,
    inplace: bool = False,
) -> scipy.sparse.spmatrix:
    """Computes HAL common windows ratio (CWR) score for co-occurrings terms.

    Note: The `nw_xy` is a *document-term matrix* (DTM) gives the common window count CW for terms x and y
    for each document. Since it is a DTM (not a TTM), each "term" in `nw_xy` corresponds to a co-occurring
    pair "x/y" where x and y are terms in the source corpus. The mapping between token "x/y" (in the co-occurrence
    corpus vocubulary) and the corresponding tokens "x" and "y" (in the source corpus vocabulary) is given by `vocab_mapping".

    The matrix `nw_x` gives the window counts for (single) terms of the source corpus vocabulary.

    The formula for CWR is:

        cwr_xy = nw_xy / (nw_x + nw_y - nw_xy)

    The mapping is needed since "xy" (the pair), and "x" and "y" (single tokens) are from different vocabularies.
    In short, given column "xy" in nw_xr, we need to find columns "x" and "y" in nw_x.


    Args:
        nw_xy (scipy.sparse.spmatrix): co-occurrence matrix in the form of a vectorized corpus
        nw_x (scipy.sparse.spmatrix): [description]
        vocab_mapping (dict): [description]

    Returns:
        scipy.sparse.spmatrix: [description]
    """

    nw_xy = (nw_xy if inplace else nw_xy.copy()).astype(np.float)

    w1_indicies = [vocab_mapping[i][0] for i in range(0, nw_xy.shape[1])]
    w2_indicies = [vocab_mapping[i][1] for i in range(0, nw_xy.shape[1])]

    for d in range(0, nw_xy.shape[0]):
        nw_xy[d, :] = nw_xy[d, :] / (nw_x[d, [w1_indicies]] + nw_x[d, [w2_indicies]] - nw_xy[d, :])

    nw_xy.data[np.isnan(nw_xy.data)] = 0.0
    nw_xy.data[nw_xy.data < 0] = 0
    nw_xy.eliminate_zeros()

    return nw_xy
