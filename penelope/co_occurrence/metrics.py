import numpy as np
import scipy
import scipy.sparse as sp
from loguru import logger


def PPMI(TTM: scipy.sparse.csc_matrix) -> scipy.sparse.csc_matrix:
    """Tranform a term-term count matrix to PPMI.
        Source:
    Args:
      C: scipy.sparse.csc_matrix of counts C_ij

    Returns:
      (scipy.sparse.csc_matrix) PPMI(C) as defined above
    """
    # Total count.
    Z = float(TTM.sum())

    # Sum each row (along columns).
    Zr = np.array(TTM.sum(axis=1), dtype=np.float64).flatten()

    # Get indices of relevant elements.
    ii, jj = TTM.nonzero()  # row, column indices
    Cij = np.array(TTM[ii, jj], dtype=np.float64).flatten()

    # PMI equation.
    pmi = np.log(Cij * Z / (Zr[ii] * Zr[jj]))

    # Truncate to positive only.
    ppmi = np.maximum(0, pmi)  # take positive only

    # Re-format as sparse matrix.
    ret = scipy.sparse.csc_matrix((ppmi, (ii, jj)), shape=TTM.shape, dtype=np.float64)
    ret.eliminate_zeros()  # remove zeros
    return ret


# def to_PPMI2(mat):
#     """Computes PPMI values for the raw co-occurrence matrix."""

#     (n_rows, n_columns) = mat.shape

#     col_totals = np.zeros(n_columns, dtype=np.float64)
#     for j in range(0, n_columns):
#         col_totals[j] = np.sum(mat[:,j].data)

#     logger.info(f"Columns totals: {col_totals}")
#     N = np.sum(col_totals)
#     for i in range(0, n_rows):
#         row = mat[i,:]
#         row_totals = np.sum(row.data)
#         for j in row.indices:
#             val = np.log((mat[i,j] * N) / (row_totals * col_totals[j]))
#             mat[i,j] = max(0, val)
#     return mat


def to_PPMI3(TTM):
    """Computes PPMI values for the raw co-occurrence matrix."""

    (n_rows, n_cols) = TTM.shape

    logger.info(f"Shape: {TTM.shape}")

    col_totals = TTM.sum(axis=0)
    row_totals = TTM.sum(axis=1).T

    N = np.sum(row_totals)

    row_matrix = np.ones((n_rows, n_cols), dtype=np.float64)
    for i in range(n_rows):
        row_matrix[i, :] = 0 if row_totals[0, i] == 0 else row_matrix[i, :] * (1.0 / row_totals[0, i])

    column_matrix = np.ones((n_rows, n_cols), dtype=np.float)
    for j in range(n_cols):
        column_matrix[:, j] = 0 if col_totals[0, j] == 0 else (1.0 / col_totals[0, j])

    P = N * TTM.toarray() * row_matrix * column_matrix

    P = np.fmax(np.zeros((n_rows, n_cols), dtype=np.float64), np.log(P))
    return sp.csr_matrix(P)
