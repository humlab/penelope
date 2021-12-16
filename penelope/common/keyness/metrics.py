from enum import IntEnum, unique
from typing import Callable, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp


@unique
class KeynessMetric(IntEnum):
    TF = 0
    TF_normalized = 1
    TF_IDF = 2
    HAL_cwr = 3
    PPMI = 4
    DICE = 5
    LLR = 6
    LLR_Z = 7
    LLR_N = 8


@unique
class KeynessMetricSource(IntEnum):
    Full = 0
    Concept = 1
    Weighed = 2


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


def _pmi(*, Cij: np.ndarray, Z: float, Zr: np.ndarray, ii: np.ndarray, jj: np.ndarray, **_):
    """Computes PMI (pointwise mutual information)"""
    with np.errstate(divide='ignore'):
        values = np.log(Cij * Z / (Zr[ii] * Zr[jj]))
    return values


def _ppmi(*, Cij: np.ndarray, Z: float, Zr: np.ndarray, ii: np.ndarray, jj: np.ndarray, normalize=False, **_):
    """Computes PPMI (positive PMI)"""
    values = _pmi(Cij=Cij, Z=Z, Zr=Zr, ii=ii, jj=jj)
    if normalize:
        values = values / -np.log(Cij * Z)
    return np.maximum(0, values)


def _dice(*, Cij: np.ndarray, Z: float, Zr: np.ndarray, ii: np.ndarray, jj: np.ndarray, **_):
    """Computes DICE coefficient ratio"""
    with np.errstate(divide='ignore'):
        values = 2.0 * Cij / (Zr[ii] + Zr[jj])
    return values


def _llr(
    *, Cij: np.ndarray, Z: float, Zr: np.ndarray, ii: np.ndarray, jj: np.ndarray, K: float, normalize: bool = False, **_
):
    """
        Word association score using Dunning's [Dunning 1993] hypothesis test.
    Args:
        Cij (np.ndarray): Co-occurrence counts for (wi, wj)
        Z (float): Total number of co-occurrences
        Zr (np.ndarray): Occurrence counts for wi
        ii (np.ndarray): Word wi indices
        jj (np.ndarray): Word wi indices
        k: Number of contexts (windows, documents)???

    Returns:
        np.ndarray: Dunning's LLR score
    """

    def ln(a: np.ndarray) -> np.ndarray:
        return np.log(np.clip(a, a_min=1e-10, a_max=None))

    K = Z
    values = 2.0 * (
        (K * ln(K))
        - (Zr[ii] * ln(Zr[ii]))
        - (Zr[jj] * ln(Zr[jj]))
        + (Cij * ln(Cij))
        + (K - Zr[ii] - Zr[jj] + Cij) * ln(K - Zr[ii] - Zr[jj] + Cij)
        + (Zr[ii] - Cij) * ln(Zr[ii] - Cij)
        + (Zr[jj] - Cij) * ln(Zr[jj] - Cij)
        - (K - Zr[ii]) * ln(K - Zr[ii])
        - (K - Zr[jj]) * ln(K - Zr[jj])
    )
    return values


def _llr_dunning(*, Cij: np.ndarray, Z: float, Zr: np.ndarray, ii: np.ndarray, jj: np.ndarray, **_) -> np.ndarray:
    """
            This code is a modified version of https://github.com/amueller/word_cloud (c)

            Word association score using Dunning's [Dunning 1993] hypothesis test.

        Args:
            Cij (np.ndarray): Co-occurrence counts for (wi, wj)
            Z (float): Total number of co-occurrences (or words?)

            Zr (np.ndarray): Occurrence counts for wi
            ii (np.ndarray): Word wi indices
            jj (np.ndarray): Word wi indices


    log-likelihood_dunning =

                   L(O11; C1; r) · L(O12; C2; r)
    LLR  = −2 log --------------------------------
                  L(O11; C1; r1) · L(O12; C2; r2)

    L(k; n; r) = pow(r, k) ∙ pow(1 − r, n − k)
            r  = R1 / N
           r1  = O11 / C1
           r2  = O12 / C2

        Returns:
            np.ndarray: Dunning's LLR score
    """

    def l(k: np.ndarray, n: np.ndarray, x: np.ndarray) -> np.ndarray:  # noqa: E741, E743
        # dunning's likelihood ratio with notation from
        # http://nlp.stanford.edu/fsnlp/promo/colloc.pdf p162
        # np.log(max(x, 1e-10)) * k + np.log(max(1 - x, 1e-10)) * (n - k)
        return np.log(np.clip(x, a_min=1e-10, a_max=None)) * k + np.log(np.clip(1 - x, a_min=1e-10, a_max=None)) * (
            n - k
        )

    P = Zr[jj] / Z
    Pi = Cij / Zr[ii]
    Pj = (Zr[jj] - Cij) / (Z - Zr[ii])

    return (
        -2.0 * l(Cij, Zr[ii], P) + l(Zr[jj] - Cij, Z - Zr[ii], P) - l(Cij, Zr[ii], Pi) - l(Zr[jj] - Cij, Z - Zr[ii], Pj)
    )


def _llr_dunning_colloc(
    *, Cij: np.ndarray, Z: float, Zr: np.ndarray, ii: np.ndarray, jj: np.ndarray, **_
) -> np.ndarray:
    """

    http://tdunning.blogspot.com/2008/03/surprise-and-coincidence.html

            This code is a modified version of https://github.com/amueller/word_cloud (c)

            Word association score using Dunning's [Dunning 1993] hypothesis test.

        Args:
            Cij (np.ndarray): Co-occurrence counts for (wi, wj)
            Z (float): Total number of co-occurrences (or words?)

            Zr (np.ndarray): Occurrence counts for wi
            ii (np.ndarray): Word wi indices
            jj (np.ndarray): Word wi indices


    log-likelihood_dunning =

                   L(O11; C1; r) · L(O12; C2; r)
    LLR  = −2 log --------------------------------
                  L(O11; C1; r1) · L(O12; C2; r2)

    L(k; n; r) = pow(r, k) ∙ pow(1 − r, n − k)
            r  = R1 / N
           r1  = O11 / C1
           r2  = O12 / C2

        Returns:
            np.ndarray: Dunning's LLR score
    """

    def l(k: np.ndarray, n: np.ndarray, x: np.ndarray) -> np.ndarray:  # noqa: E741, E743
        # dunning's likelihood ratio with notation from
        # http://nlp.stanford.edu/fsnlp/promo/colloc.pdf p162
        # np.log(max(x, 1e-10)) * k + np.log(max(1 - x, 1e-10)) * (n - k)
        return np.log(np.clip(x, a_min=1e-10, a_max=None)) * k + np.log(np.clip(1 - x, a_min=1e-10, a_max=None)) * (
            n - k
        )

    P = Zr[jj] / Z
    Pi = Cij / Zr[ii]
    Pj = (Zr[jj] - Cij) / (Z - Zr[ii])

    return (
        -2.0 * l(Cij, Zr[ii], P) + l(Zr[jj] - Cij, Z - Zr[ii], P) - l(Cij, Zr[ii], Pi) - l(Zr[jj] - Cij, Z - Zr[ii], Pj)
    )


# def _nltk_col_log_likelihood(count_a, count_b, count_ab, N):
#     """

#     https://www.nltk.org/_modules/nltk/tokenize/punkt.html

#     A function that will just compute log-likelihood estimate, in
#     the original paper it's decribed in algorithm 6 and 7.
#     This *should* be the original Dunning log-likelihood values,
#     unlike the previous log_l function where it used modified
#     Dunning log-likelihood values
#     """
#     import math

#     p = 1.0 * count_b / N
#     p1 = 1.0 * count_ab / count_a
#     p2 = 1.0 * (count_b - count_ab) / (N - count_a)

#     summand1 = count_ab * math.log(p) + (count_a - count_ab) * math.log(1.0 - p)

#     summand2 = (count_b - count_ab) * math.log(p) + (N - count_a - count_b + count_ab) * math.log(1.0 - p)

#     if count_a == count_ab:
#         summand3 = 0
#     else:
#         summand3 = count_ab * math.log(p1) + (count_a - count_ab) * math.log(1.0 - p1)

#     if count_b == count_ab:
#         summand4 = 0
#     else:
#         summand4 = (count_b - count_ab) * math.log(p2) + (N - count_a - count_b + count_ab) * math.log(1.0 - p2)

#     likelihood = summand1 + summand2 - summand3 - summand4

#     return -2.0 * likelihood


# https://github.com/DrDub/icsisumm/blob/1cb583f86dddd65bfeec7bb9936c97561fd7811b/icsisumm-primary-sys34_v1/nltk/nltk-0.9.2/nltk/tokenize/punkt.py


def _llr_dunning_n_words(
    *, Cij: np.ndarray, Z: float, Zr: np.ndarray, ii: np.ndarray, jj: np.ndarray, N: float, **_
) -> np.ndarray:
    return _llr_dunning(Cij=Cij, Z=N, Zr=Zr, ii=ii, jj=jj)


# ### Evaluate the "surprise factor" of two proportions that are expressed as counts.
# ###  ie x1 "heads" out of n1 flips.
# def dunning_score(x1, x2, n1, n2):
#     p1 = float(x1) / n1
#     p2 = float(x2) / n2
#     p = float(x1 + x2) / (n1 + n2)

#     return -2 * ( x1 * math.log(p / p1) + (n1 - x1) * math.log((1 - p)/(1 - p1)) +
#                   x2 * math.log(p / p2) + (n2 - x2) * math.log((1 - p)/(1 - p2)) )


# def dunning_log_likelihood(f1, s1, f2, s2):
#     """Calculates Dunning log likelihood of an observation in two groups.
#     This determines if an observation is more strongly associated with
#     one of two groups, and the strength of that association.
#     Args:
#         f1: Integer, observation frequency in group one.
#         s1: Integer, total data points in group one.
#         f2: Integer, observation frequency in group two.
#         s2: Integer, total data points in group two.
#     Returns:
#         Float log likelihood. This will be positive if the
#         observation is more likely in group one. More extreme
#         values indicate a stronger association.
#     """
#     if f1 + f2 == 0:
#         return 0.0
#     if s1 == 0 or s2 == 0:
#         return 0.0
#     f1, s1, f2, s2 = float(f1), float(s1), float(f2), float(s2)
#     # Expected values
#     e1 = s1 * (f1 + f2) / (s1 + s2)
#     e2 = s2 * (f1 + f2) / (s1 + s2)
#     l1, l2 = 0, 0
#     if e1 != 0 and f1 != 0:
#         l1 = f1 * math.log(f1 / e1)
#     if e2 != 0 and f2 != 0:
#         l2 = f2 * math.log(f2 / e2)

#     likelihood = 2 * (l1 + l2)
#     if f2 / s2 > f1 / s1:
#         likelihood = -likelihood
#     return likelihood


def _undefined(Cij, Z, Zr, ii, jj, k, *_):
    """Computes log-likelihood ratio"""
    raise ValueError("metric is not applicable in current context")


def _hal_cwr(*, Cij: np.ndarray, Z: float, Zr: np.ndarray, ii: np.ndarray, jj: np.ndarray, N: float, **_) -> np.ndarray:
    """Computes HAL common windows ratio (CWR) score for co-occurrings terms.
    The formula for CWR is:

        cwr_xy = nw_xy / (nw_x + nw_y - nw_xy)

    """

    nw_xy = Cij
    nw_x = Zr[ii]
    nw_y = Zr[jj]

    with np.errstate(divide='ignore'):
        score = nw_xy / (nw_x + nw_y - nw_xy)

    return score


METRIC_FUNCTION = {
    KeynessMetric.PPMI: _ppmi,
    KeynessMetric.DICE: _dice,
    KeynessMetric.LLR: _llr,
    KeynessMetric.LLR_Z: _llr_dunning,
    KeynessMetric.LLR_N: _llr_dunning_n_words,
    KeynessMetric.HAL_cwr: _hal_cwr,
}


def significance(
    TTM: sp.csc_matrix,
    metric: Union[Callable, KeynessMetric],
    normalize: bool = False,
    n_contexts=None,
    n_words=None,
) -> sp.csc_matrix:
    """Computes statistical significance tf co-occurrences using `metric`.

    Args:
        TTM (sp.csc_matrix): [description]
        normalize (bool, optional): [description]. Defaults to False.

    Returns:
        sp.csc_matrix: [description]
    """
    metric = metric if callable(metric) else METRIC_FUNCTION.get(metric, _undefined)

    K: float = n_contexts
    N: float = n_words

    """Total number of observations (counts)"""
    Z: float = float(TTM.sum())

    """Number of observations per context (document, row sum)"""
    Zr = np.array(TTM.sum(axis=1), dtype=np.float64).flatten()

    """Row and column indices of non-zero elements."""
    ii, jj = TTM.nonzero()

    Cij: np.ndarray = np.array(TTM[ii, jj], dtype=np.float64).flatten()

    """Compute weights (with optional normalize)."""
    weights: np.ndarray = metric(Cij=Cij, Z=Z, Zr=Zr, ii=ii, jj=jj, K=K, N=N, normalize=normalize)

    np.nan_to_num(
        weights,
        copy=False,
        posinf=0.0,
        neginf=0.0,
        nan=0.0,
    )

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
    document_index: pd.DataFrame = None,
    vocabulary_size: int = None,
    normalize: bool = False,
) -> pd.DataFrame:
    """Computes keyness values for `co-occurrences` data frame

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
        pivot_co_occurrences = co_occurrences[co_occurrences[pivot_key] == period]
        term_term_matrix = sp.csc_matrix(
            (pivot_co_occurrences.value, (pivot_co_occurrences.w1_id, pivot_co_occurrences.w2_id)),
            shape=(vocabulary_size, vocabulary_size),
            dtype=np.float64,
        )
        n_contexts = _get_documents_count(document_index, pivot_co_occurrences)
        weights, (w1_ids, w2_ids) = significance(
            TTM=term_term_matrix,
            metric=keyness_metric,
            normalize=normalize,
            n_contexts=n_contexts,
        )
        co_occurrence_partitions.append(
            pd.DataFrame(data={pivot_key: period, 'w1_id': w1_ids, 'w2_id': w2_ids, 'value': weights})
        )
    keyness_co_occurrences = pd.concat(co_occurrence_partitions, ignore_index=True)
    return keyness_co_occurrences


def _get_documents_count(document_index: pd.DataFrame, co_occurrences: pd.DataFrame) -> int:
    """Returns number of documents that has contributed to data in `co_occurrences`"""
    if 'document_id' not in co_occurrences.columns:
        raise ValueError("fatal: document index has no ID column")
    documents_ids: Sequence[int] = co_occurrences.document_id.unique()
    if len(documents_ids) == 0:
        return 0
    if 'n_documents' in document_index.columns:
        n_contexts = document_index.loc[documents_ids].n_documents.sum()
    else:
        n_contexts = len(document_index.loc[documents_ids])
    return n_contexts


def significance_ratio(A: sp.spmatrix, B: sp.spmatrix) -> sp.spmatrix:
    """https://stackoverflow.com/posts/58446948"""
    inv_B: sp.spmatrix = B.copy()
    inv_B.data = 1 / inv_B.data
    return A.multiply(inv_B)
