import numpy as np
import pandas as pd
import scipy.sparse as sp

# @deprecated
# def compute_hal_score_cellwise(nw_xy: scipy.sparse.spmatrix, nw_x: scipy.sparse.spmatrix, vocabs_mapping: dict) -> scipy.sparse.spmatrix:
#     # Cell by CELL
#     for d in range(0, nw_xy.shape[0]):
#         for t in range(0, nw_xy.shape[1]):
#             w1_id = vocabs_mapping[t][0]
#             w2_id = vocabs_mapping[t][1]
#             nw_xy[d, t] = nw_xy[d, t] / (nw_x[d, w1_id] + nw_x[d, w2_id] - nw_xy[d, t])

#     return nw_xy

# @deprecated
# def compute_hal_score_colwise(nw_xy: scipy.sparse.spmatrix, nw_x: scipy.sparse.spmatrix, vocabs_mapping: dict) -> scipy.sparse.spmatrix:
#     # COL by COL
#     for t in range(0, nw_xy.shape[1]):
#         nw_xy[:, t] = nw_xy[:, t]  / (nw_x[:,vocabs_mapping[t][0]] + nw_x[:,vocabs_mapping[t][1]] - nw_xy[:, t])
#     return nw_xy

SparseMatrix = sp.spmatrix


def compute_hal_score_by_co_occurrence_matrix(co_occurrences: pd.DataFrame, nw_x_matrix: SparseMatrix) -> pd.Series:
    """Compute yearly HAL-score for each co-occurrence pair (w1, w2)"""

    # nw_xy is given by co_occurrences data frame
    co_occurrences = co_occurrences[['document_id', 'w1_id', 'w2_id', 'value']]
    co_occurrences['nw_xy'] = co_occurrences.value

    nw_x_matrix: SparseMatrix = nw_x_matrix.tocoo()
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
    nw_xy: SparseMatrix,
    nw_x: SparseMatrix,
    vocabs_mapping: dict,
    inplace: bool = False,
) -> SparseMatrix:
    """Computes HAL common windows ratio (CWR) score for co-occurrings terms.

    Note: The `nw_xy` is a *document-term matrix* (DTM) gives the common window count CW for terms x and y
    for each document. Since it is a DTM (not a TTM), each "term" in `nw_xy` corresponds to a co-occurring
    pair "x/y" where x and y are terms in the source corpus. The mapping between token "x/y" (in the co-occurrence
    corpus vocubulary) and the corresponding tokens "x" and "y" (in the source corpus vocabulary) is given by `vocabs_mapping".

    The matrix `nw_x` gives the window counts for (single) terms of the source corpus vocabulary.

    The formula for CWR is:

        cwr_xy = nw_xy / (nw_x + nw_y - nw_xy)

    The mapping is needed since "xy" (the pair), and "x" and "y" (single tokens) are from different vocabularies.
    In short, given column "xy" in nw_xr, we need to find columns "x" and "y" in nw_x.


    Args:
        nw_xy (SparseMatrix): co-occurrence matrix in the form of a vectorized corpus
        nw_x (SparseMatrix): [description]
        vocabs_mapping (dict): [description]

    Returns:
        SparseMatrix: [description]
    """

    nw_xy = (nw_xy if inplace else nw_xy.copy()).astype(np.float64)

    reverse_mapping = {v: k for k, v in vocabs_mapping.items()}

    w1_indices = [reverse_mapping[i][0] for i in range(0, nw_xy.shape[1])]
    w2_indices = [reverse_mapping[i][1] for i in range(0, nw_xy.shape[1])]

    for d in range(0, nw_xy.shape[0]):
        nw_xy[d, :] = nw_xy[d, :] / (nw_x[d, w1_indices] + nw_x[d, w2_indices] - nw_xy[d, :])

    nw_xy.data[np.isnan(nw_xy.data)] = 0.0
    nw_xy.data[nw_xy.data < 0] = 0
    nw_xy.eliminate_zeros()

    return nw_xy

    # # nw_xy = (nw_xy if inplace else nw_xy.copy()).astype(np.float64)

    # reverse_mapping = {v: k for k, v in vocabs_mapping.items()}

    # w1_indices = [reverse_mapping[i][0] for i in range(0, nw_xy.shape[1])]
    # w2_indices = [reverse_mapping[i][1] for i in range(0, nw_xy.shape[1])]

    # out_nw_xy: SparseMatrix = sp.lil_matrix(nw_xy.shape, dtype=np.float64)

    # for d in range(0, nw_xy.shape[0]):
    #     # nw_xy[d, :] = nw_xy[d, :] / (nw_x[d, w1_indices] + nw_x[d, w2_indices] - nw_xy[d, :])
    #     out_nw_xy[d,:] = nw_xy[d, :] / (nw_x[d, w1_indices] + nw_x[d, w2_indices] - nw_xy[d, :])

    # out_nw_xy.data[np.isnan(out_nw_xy.data)] = 0.0
    # out_nw_xy.data[out_nw_xy.data < 0] = 0
    # out_nw_xy.eliminate_zeros()

    # return out_nw_xy.tocsr()
