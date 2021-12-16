import math

# from numpy.typing import ArrayLike
import warnings
from typing import Any, Tuple

import numpy as np
import scipy
from numpy.polynomial.polynomial import polyfit
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning, module='numpy.polynomial.polynomial')


def gof_by_l2_norm(matrix: scipy.sparse.spmatrix, axis: int = 1, scale: bool = True):

    """Computes L2 norm for rows (axis = 1) or columns (axis = 0).

    See stats.stackexchange.com/questions/25827/how-does-one-measure-the-non-uniformity-of-a-distribution

    Measures distance tp unform distribution (1/sqrt(d))

    "The lower bound 1d√ corresponds to uniformity and upper bound to the 1-hot vector."

    "It just so happens, though, that the L2 norm has a simple algebraic connection to the χ2 statistic used in goodness of fit tests:
    that's the reason it might be suitable to measure non-uniformity"

    """
    d = matrix.shape[axis]  # int(not axis)]

    if scipy.sparse.issparse(matrix):
        matrix = matrix.todense()

    l2_norm = np.linalg.norm(matrix, axis=axis)

    if scale:
        l2_norm = (l2_norm * math.sqrt(d) - 1) / (math.sqrt(d) - 1)

    return l2_norm


Point = Tuple[float, float]


def fit_ordinary_least_square(ys, xs=None) -> Tuple[float, float, Any, Point, Point]:
    """[summary]

    Parameters
    ----------
    ys : array-like
        observations
    xs : array-like, optional
        categories, domain, by default None

    Returns
    -------
    tuple m, k, p, (x1, x2), (y1, y2)
        where y = k * x + m
          and p is the p-value
    """
    if xs is None:
        xs = np.arange(len(ys))  # 0.0, len(ys), 1.0))

    slope, intercept, _, p, _ = stats.linregress(xs, ys)

    (x1, x2), (y1, y2) = (xs[0], xs[-1]), (intercept + slope * xs[0], intercept + slope * xs[-1])

    return intercept, slope, p, (x1, x2), (y1, y2)


def fit_ordinary_least_square_ravel(Y, xs):

    if xs is None:
        xs = np.arange(Y.shape[0])

    xsr = np.repeat(xs, Y.shape[1])
    ysr = Y.ravel()

    return fit_ordinary_least_square(ys=ysr, xs=xsr)


def fit_polynomial(xs, ys, deg=1):
    return polyfit(xs, ys, deg)


def fit_polynomial_ravel(Y, xs):
    """Layout columns as a single y-vector using ravel. Repeat x-vector for each column"""
    xsr = np.repeat(xs, Y.shape[1])
    ysr = Y.ravel()
    kx_m = np.polyfit(x=xsr, y=ysr, deg=1)
    return kx_m


def gof_chisquare_to_uniform(f_obs, axis=0):

    (chi2, p) = scipy.stats.chisquare(f_obs, axis=axis)

    return (chi2, p)


def earth_mover_distance(vs, ws=None):

    if ws is None:
        ws = np.full(len(vs), vs.mean())

    return scipy.stats.wasserstein_distance(vs, ws)


def entropy(pk, qk=None):

    if qk is None:
        qk = np.full(len(pk), pk.mean())

    return scipy.stats.entropy(pk, qk)


def kullback_leibler_divergence(p, q):
    if q is None:
        q = np.full(len(p), p.mean())
    e = 0.00001  # avoid div by zero
    kld = np.sum((p + e) * np.log((p + e) / (q + e)))
    return kld


def kullback_leibler_divergence_to_uniform(p):
    q = p.mean()
    e = 0.00001  # avoid div by zero
    kld = np.sum((p + e) * np.log((p + e) / (q + e)))
    return kld
