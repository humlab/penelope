import numpy as np
import pandas as pd
import scipy
import scipy.optimize
from numpy.polynomial.polynomial import polyval


def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(-(((x - mean) / standard_deviation) ** 2))


def linear(x, a, b):
    return polyval(x, [a, b])


def polynomial2(x, a, b, c):
    return polyval(x, [a, b, c])


def polynomial3(x, a, b, c, d):
    return polyval(x, [a, b, c, d])


def polynomial4(x, a, b, c, d, e):
    return polyval(x, [a, b, c, d, e])


def sigmoid(x, x0, k):
    y = 1 / (1 + np.exp(-k * (x - x0)))
    return y


def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c


def zipf_mandelbrot(x, a, b, c):
    return c / (x + b) ** a


def inversed(x, a, b):
    return a / (x - b)


def square_root(x, a, b, c):
    return a * np.sqrt(x / c - b)


def pchip_spline(xs, ys):
    # FIXME: #21 PchipInterpolator ValueError: setting an array element with a sequence.
    if isinstance(ys, scipy.sparse.spmatrix):
        ys = ys.todense().A1
    spliner = scipy.interpolate.PchipInterpolator(xs, ys)
    s_xs = np.arange(xs.min(), xs.max() + 0.1, 0.1)
    s_ys = spliner(s_xs)
    return s_xs, s_ys


def pchip_spliner(xs, ys):
    if isinstance(ys, scipy.sparse.spmatrix):
        ys = ys.todense().A1
    spliner = scipy.interpolate.PchipInterpolator(xs, ys)
    return spliner


def fit_curve(fx, xs, ys, step=0.1):
    popt, _ = scipy.optimize.curve_fit(fx, xs, ys)  # pylint: disable=unbalanced-tuple-unpacking
    s_xs = np.arange(xs.min(), xs.max() + step, step)
    s_ys = fx(s_xs, *popt)
    return s_xs, s_ys


fit_polynomial2 = lambda xs, ys: fit_curve(polynomial2, xs, ys)
fit_polynomial3 = lambda xs, ys: fit_curve(polynomial3, xs, ys)
fit_polynomial4 = lambda xs, ys: fit_curve(polynomial4, xs, ys)


def fit_curve_ravel(fx, xs, Y, step=0.1):

    assert len(Y.shape) == 2

    if xs is None:
        xs = np.arange(Y.shape[0])

    xsr = np.repeat(xs, Y.shape[1])
    ysr = Y.ravel()

    return fit_curve(fx, xsr, ysr, step=step)


rolling_average_methods = {
    'pandas': lambda xs, ys, n: (xs, pd.Series(ys).rolling(window=n).mean().iloc[n - 1 :].values),
    'fftconvolve': lambda xs, ys, n: (xs, scipy.signal.fftconvolve(ys, np.ones((n,)) / n, mode='valid')),
    'reflect': lambda xs, ys, n: (xs, scipy.ndimage.filters.uniform_filter1d(ys, size=n, mode='reflect')),
    'nearest': lambda xs, ys, n: (xs, scipy.ndimage.filters.uniform_filter1d(ys, size=n, mode='nearest')),
    'mirror': lambda xs, ys, n: (xs, scipy.ndimage.filters.uniform_filter1d(ys, size=n, mode='mirror')),
    'constant': lambda xs, ys, n: (xs, scipy.ndimage.filters.uniform_filter1d(ys, size=n, mode='constant')),
}


def rolling_average_smoother(key, window_size):
    def smoother(xs, ys):
        xs, ys = rolling_average_methods[key](xs, ys, window_size)
        ys = ys / ys.sum()
        return xs, ys

    return smoother


def boxplot_statistics(values):

    if len(values) == 0:
        return 0, 0, 0, 0, 0, np.empty(shape=(0, 0))

    q1, q2, q3 = np.percentile(values, q=(25, 50, 75))

    # In descriptive statistics, the interquartile range (iqr), also called the midspread or middle 50%,
    # or technically H-spread, is a measure of statistical dispersion, being equal to the difference
    # between 75th and 25th percentiles (https://en.wikipedia.org/wiki/Interquartile_range)

    iqr = q3 - q1

    upper_whisker = values[values <= q3 + 1.5 * iqr].max()
    lower_whisker = values[values >= q1 - 1.5 * iqr].min()

    outliers = values[(values > upper_whisker) | (values < lower_whisker)]

    return (q1, q2, q3), iqr, (upper_whisker, lower_whisker), outliers
