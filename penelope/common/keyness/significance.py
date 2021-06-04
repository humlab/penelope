from enum import IntEnum
from typing import Callable, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp


class KeynessMetric(IntEnum):
    TF = 0
    TF_normalized = 1
    TF_IDF = 2
    HAL_cwr = 3
    PPMI = 4
    DICE = 5
    LLR = 6

"""Computes statistical significances of co-occurrences

### REASONING

FIXME: Add source for this (Dunning?):

Raw popularity count is too crude of a measure. We have to find more clever statistics to be able to pick out
meaningful phrases easily. For a given pair of words, the method tests two hypotheses on the observed dataset.
- Hypothesis 1 (the null hypothesis) says that word 1 appears independently from word 2.
- Hypothesis 2 (the alternate hypothesis) says that seeing word 1 changes the likelihood of seeing word 2.

Hence, the likelihood ratio test (LLR) for phrase detection (a.k.a. collocation extraction) asks the following question:
Are the observed word occurrences in a given text corpus more likely to have been generated from a model where
the two words occur independently from one another?


SOURCE: https://tm4ss.github.io/docs/Tutorial_5_Co-occurrence.html

######### PMI: log(k*kij / (ki * kj) ########
PMI_significance = log(k * kij / (ki * kj))

########## DICE: 2 X&Y / X + Y ##############
DICE_significance = 2 * kij / (ki + kj)

########## Log Likelihood ###################
LLR_significance = 2 * ((k * log(k)) - (ki * log(ki)) - (kj * log(kj)) + (kij * log(kij))
               + (k - ki - kj + kij) * log(k - ki - kj + kij)
               + (ki - kij) * log(ki - kij) + (kj - kij) * log(kj - kij)
               - (k - ki) * log(k - ki) - (k - kj) * log(k - kj))

"""

# pylint: disable=unused-argument

# FIXME: How to normalize?


def _pmi(Cij, Z, Zr, ii, jj, *_, normalize=False):
    """Computes PMI (pointwise mutual information)"""
    values = np.log(Cij * Z / (Zr[ii] * Zr[jj]))
    return values


def _ppmi(Cij, Z, Zr, ii, jj, *_, normalize=False):
    """Computes PPMI (positive PMI)"""
    values = _pmi(Cij, Z, Zr, ii, jj)
    if normalize:
        values = values / -np.log(Cij * Z)
    return np.maximum(0, values)


def _dice(Cij, Z, Zr, ii, jj, *_, normalize=False):
    """Computes DICE coefficient ratio"""
    values = 2.0 * Cij / (Zr[ii] + Zr[jj])
    return values


def _llr(Cij, Z, Zr, ii, jj, k, *_, normalize=False):
    """Computes log-likelihood ratio"""
    values = 2.0 * (
        (k * np.log(k))
        - (Zr[ii] * np.log(Zr[ii]))
        - (Zr[jj] * np.log(Zr[jj]))
        + (Cij * np.log(Cij))
        + (k - Zr[ii] - Zr[jj] + Cij) * np.log(k - Zr[ii] - Zr[jj] + Cij)
        + (Zr[ii] - Cij) * np.log(Zr[ii] - Cij)
        + (Zr[jj] - Cij) * np.log(Zr[jj] - Cij)
        - (k - Zr[ii]) * np.log(k - Zr[ii])
        - (k - Zr[jj]) * np.log(k - Zr[jj])
    )
    return values


def _llr_dunning(Cij, Z, Zr, ii, jj, *_):
    def l(k, n, x):  # noqa: E741, E743
        # dunning's likelihood ratio with notation from
        # http://nlp.stanford.edu/fsnlp/promo/colloc.pdf p162
        return np.log(max(x, 1e-10)) * k + np.log(max(1 - x, 1e-10)) * (n - k)

    p = Zr[jj] / Z
    Pi = Cij / Zr[ii]
    Pj = (Zr[jj] - Cij) / (Z - Zr[ii])

    score = l(Cij, Zr[ii], p) + l(Zr[jj] - Cij, Z - Zr[ii], p) - l(Cij, Zr[ii], Pi) - l(Zr[jj] - Cij, Z - Zr[ii], Pj)

    return -2.0 * score


def _llr_not_used(k, K, n, N):
    """
    Compute the Log Likelihood Ratio.
    """
    val = k * np.log(float(k * N) / float(n * K))
    if n > k:
        val += (n - k) * np.log(float((n - k) * N) / float(n * (N - K)))
    if K > k:
        val += (K - k) * np.log(float(N * (K - k)) / float(K * (N - n)))
    if (N - K - n + k) > 0:
        val += (N - K - n + k) * np.log(float(N * (N - K - n + k)) / float((N - K) * (N - n)))
    return val


def _undefined(Cij, Z, Zr, ii, jj, k, *_):
    """Computes log-likelihood ratio"""
    raise ValueError("metric is not applicable in current context")


METRIC_FUNCTION = {
    KeynessMetric.PPMI: _ppmi,
    KeynessMetric.DICE: _dice,
    KeynessMetric.LLR: _llr,
}


def significance(TTM: sp.csc_matrix, metric: Union[Callable, KeynessMetric], normalize: bool = False) -> sp.csc_matrix:
    """Computes statistical significance if co-occurrences using `metric`.

    Args:
        TTM (sp.csc_matrix): [description]
        normalize (bool, optional): [description]. Defaults to False.

    Returns:
        sp.csc_matrix: [description]
    """
    metric = metric if callable(metric) else METRIC_FUNCTION.get(metric, _undefined)

    # Number of contexts (documents)
    k = TTM.shape[0]  # FIXME: This is wrong since TTM is a term-term matrix!!!

    # Total number of observations (counts)
    Z: float = float(TTM.sum())  # Total number of observations (counts)

    # Number of observations per context (document, row sum)
    Zr = np.array(TTM.sum(axis=1), dtype=np.float64).flatten()

    # Row and column indices of non-zero elements.
    ii, jj = TTM.nonzero()

    Cij: np.ndarray = np.array(TTM[ii, jj], dtype=np.float64).flatten()

    # Compute weights (with optional normalize).
    weights: np.ndarray = metric(Cij, Z, Zr, ii, jj, k)

    weights = np.nan_to_num(weights, posinf=0.0, neginf=0.0, nan=0.0)

    nz_indices: np.ndarray = weights.nonzero()

    return (weights[nz_indices], (ii[nz_indices], jj[nz_indices]))


def significance_matrix(
    TTM: sp.csc_matrix, metric: Union[Callable, KeynessMetric], normalize: bool = False
) -> sp.csc_matrix:

    weights, (ii, jj) = significance(TTM, metric, normalize)

    M: sp.spmatrix = sp.csc_matrix((weights, (ii, jj)), shape=TTM.shape, dtype=np.float64)
    M.eliminate_zeros()

    return M


def partitioned_significances(
    co_occurrences: pd.DataFrame,
    pivot_key: str,
    keyness_metric: KeynessMetric,
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
    co_occurrence_partitions = []
    for period in co_occurrences[pivot_key].unique():
        co_partition = co_occurrences[co_occurrences[pivot_key] == period]
        ttm = sp.csc_matrix(
            (co_partition.value, (co_partition.w1_id, co_partition.w2_id)),
            shape=(vocabulary_size, vocabulary_size),
            dtype=np.float64,
        )
        weights, (w1_ids, w2_ids) = significance(TTM=ttm, metric=keyness_metric, normalize=normalize)
        co_occurrence_partitions.append(
            pd.DataFrame(data={pivot_key: period, 'w1_id': w1_ids, 'w2_id': w2_ids, 'value': weights})
        )
    weighed_co_occurrences = pd.concat(co_occurrence_partitions, ignore_index=True)
    return weighed_co_occurrences
