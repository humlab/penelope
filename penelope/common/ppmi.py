import numpy as np
import pandas as pd
import scipy.sparse as sp


def PPMI(TTM: sp.csc_matrix, normalize: bool = False) -> sp.csc_matrix:
    """Tranform a term-term count matrix to PPMI.

    Args:
        TTM (sp.csc_matrix): [description]
        normalize (bool, optional): [description]. Defaults to False.

    Returns:
        sp.csc_matrix: [description]

    FIXME: Add reference to original source (colaboratory notebook)
    """

    # Get total counts (Z) and row counts (Zr).
    Z: float = float(TTM.sum())

    Zr = np.array(TTM.sum(axis=1), dtype=np.float64).flatten()

    # Get row, column indices of non-zero elements.
    ii, jj = TTM.nonzero()

    Cij: np.ndarray = np.array(TTM[ii, jj], dtype=np.float64).flatten()

    # Compute PMI (with optional normalize).
    pmi: np.ndarray = np.log(Cij * Z / (Zr[ii] * Zr[jj]))

    if normalize:
        pmi = pmi / -np.log(Cij * Z)

    # Clamp values to positive only (PMI => PPMI).
    ppmi = np.maximum(0, pmi)

    # Create sparse matrix and eliminate series
    M: sp.spmatrix = sp.csc_matrix((ppmi, (ii, jj)), shape=TTM.shape, dtype=np.float64)

    M.eliminate_zeros()

    return M


def compute_partitioned_PPMI(
    co_occurrences: pd.DataFrame,
    pivot_key: str,
    vocabulary_size: int = None,
    normalize: bool = False,
) -> pd.DataFrame:
    """Computes PPMI values for `co-occurrences` data frame

    Args:
        co_occurrences (pd.DataFrame): Must have columns [ "w1_id", "w2_id". "value" ]
        pivot_key (str): Pivot key (column) in document index that specifies partitions
        shape (Tuple[int,int]): Shape

    Returns:
        pd.DataFrame: [description]
    """
    vocabulary_size: int = vocabulary_size or max(co_occurrences.w1_id, co_occurrences.w2_id)
    ppmi_co_occurrence_partitions = []
    for period in co_occurrences[pivot_key].unique():
        co_partition = co_occurrences[co_occurrences[pivot_key] == period]
        ttm = sp.csc_matrix(
            (co_partition.value, (co_partition.w1_id, co_partition.w2_id)),
            shape=(vocabulary_size, vocabulary_size),
            dtype=np.float32,
        )
        C_ppmi = PPMI(TTM=ttm, normalize=normalize).tocoo()
        ppmi_co_occurrence_partitions.append(
            pd.DataFrame(
                data={
                    pivot_key: period,
                    'w1_id': C_ppmi.row,
                    'w2_id': C_ppmi.col,
                    'value': C_ppmi.data,
                }
            )
        )
    ppmi_co_occurrences = pd.concat(ppmi_co_occurrence_partitions, ignore_index=True)
    return ppmi_co_occurrences
