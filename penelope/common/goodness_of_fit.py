import collections
import math
from typing import Dict

import bokeh
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from numpy.polynomial.polynomial import Polynomial as polyfit
from penelope.corpus.vectorized_corpus import VectorizedCorpus


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


def fit_ordinary_least_square(ys, xs=None):
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

    xs = sm.add_constant(xs)

    model = sm.OLS(endog=ys, exog=xs)
    result = model.fit()
    coeffs = result.params
    predicts = result.predict()
    (x1, x2), (y1, y2) = (xs[0][1], xs[-1][1]), (predicts[0], predicts[-1])

    return coeffs[0], coeffs[1], result.pvalues, (x1, x2), (y1, y2)


def fit_ordinary_least_square_ravel(Y, xs):

    if xs is None:
        xs = np.arange(Y.shape[0])

    xsr = np.repeat(xs, Y.shape[1])
    ysr = Y.ravel()

    return fit_ordinary_least_square(ys=ysr, xs=xsr)


def fit_polynomial(ys, xs=None, deg=1):

    if xs is None:
        xs = np.arange(len(ys))

    return polyfit.fit(xs, ys, deg).convert().coef


def fit_polynomial_ravel(Y, xs):

    # xs = np.arange(x_corpus.documents.year.min(), x_corpus.documents.year.max() + 1, 1)

    # Layout columns as a single y-vector using ravel. Repeat x-vector for each column
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


def compute_goddness_of_fits_to_uniform(
    x_corpus: VectorizedCorpus, n_top_count: int = None, verbose=False
) -> pd.DataFrame:
    """Returns metric of how well the token distributions fit a uniform distribution.

    Parameters
    ----------
    x_corpus : VectorizedCorpus
    n_top_count : int, optional
        Only return the `n_top_count` most frequent tokens, by default None
    verbose : bool, optional
        Reduces the number of returned columns, by default False

    Returns
    -------
    pd.DataFrame
        [description]
    """
    x_corpus = x_corpus.todense()
    xs_years = x_corpus.xs_years()

    dtm = x_corpus.data

    df_gof = pd.DataFrame(
        {
            'token': [x_corpus.id2token[i] for i in range(0, dtm.shape[1])],
            'word_count': [x_corpus.word_counts[x_corpus.id2token[i]] for i in range(0, dtm.shape[1])],
            'l2_norm': gof_by_l2_norm(dtm, axis=0),
        }
    )

    chi2_stats, chi2_p = list(zip(*[gof_chisquare_to_uniform(dtm[:, i]) for i in range(0, dtm.shape[1])]))
    ks, ms = list(zip(*[fit_polynomial(dtm[:, i], xs_years, 1) for i in range(0, dtm.shape[1])]))

    df_gof['slope'] = ks
    df_gof['intercept'] = ms

    df_gof['chi2_stats'] = chi2_stats
    df_gof['chi2_p'] = chi2_p

    df_gof['min'] = [dtm[:, i].min() for i in range(0, dtm.shape[1])]
    df_gof['max'] = [dtm[:, i].max() for i in range(0, dtm.shape[1])]
    df_gof['mean'] = [dtm[:, i].mean() for i in range(0, dtm.shape[1])]
    df_gof['var'] = [dtm[:, i].var() for i in range(0, dtm.shape[1])]

    df_gof['earth_mover'] = [earth_mover_distance(dtm[:, i]) for i in range(0, dtm.shape[1])]
    df_gof['entropy'] = [entropy(dtm[:, i]) for i in range(0, dtm.shape[1])]
    df_gof['kld'] = [kullback_leibler_divergence_to_uniform(dtm[:, i]) for i in range(0, dtm.shape[1])]

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html
    df_gof['skew'] = [scipy.stats.skew(dtm[:, i]) for i in range(0, dtm.shape[1])]

    # df['ols_m_k_p_xs_ys'] = [ gof.fit_ordinary_least_square(dtm[:,i], xs=xs_years) for i in range(0, dtm.shape[1]) ]
    # df['ols_k']           = [ m_k_p_xs_ys[1] for m_k_p_xs_ys in df.ols_m_k_p_xs_ys.values ]
    # df['ols_m']           = [ m_k_p_xs_ys[0] for m_k_p_xs_ys in df.ols_m_k_p_xs_ys.values ]

    df_gof.sort_values(['l2_norm'], ascending=False, inplace=True)

    # uniform_constant = 1.0 / math.sqrt(float(dtm.shape[0]))

    if (n_top_count or 0) > 0:
        df_gof = df_gof.nlargest(10, columns=["word_count"])

    if not verbose:
        df_gof = df_gof[
            [
                "token",
                "word_count",
                "l2_norm",
                "slope",
                "chi2_stats",
                "earth_mover",
                "kld",
                "skew",
                "entropy",
            ]
        ]

    return df_gof


def get_most_deviating_words(df_gof, metric, n_count=500, ascending=False, abs_value=False):

    # better sorting: df.iloc[df['b'].abs().argsort()]
    # descending: df.iloc[(-df['b'].abs()).argsort()]

    # df = (
    #     df_gof.reindex(df_gof[metric].abs().sort_values(ascending=False).index)
    #     if abs_value
    #     else df_gof.nlargest(n_count, columns=metric).sort_values(by=metric, ascending=ascending)
    # )

    # df = df.reset_index()[['token', metric]].rename(columns={'token': metric + '_token'})

    abs_metric = f'abs_{metric}'
    df = df_gof[['token', metric]].rename(columns={'token': metric + '_token'})
    df[abs_metric] = df[metric].abs()

    sort_column = abs_metric if abs_value else metric

    df = df.sort_values(by=sort_column, ascending=ascending)

    if n_count > 0:
        df = df.nlargest(n_count, columns=abs_metric)

    return df


def compile_most_deviating_words(df, n_count=500):

    xf = (
        get_most_deviating_words(df, 'l2_norm', n_count)
        .join(get_most_deviating_words(df, 'slope', n_count, abs_value=True))
        .join(get_most_deviating_words(df, 'chi2_stats', n_count))
        .join(get_most_deviating_words(df, 'earth_mover', n_count))
        .join(get_most_deviating_words(df, 'kld', n_count))
        .join(get_most_deviating_words(df, 'skew', n_count))
    )

    return xf


def plot_metric_histogram(df_gof, metric='l2_norm', bins=100):

    p = bokeh.plotting.figure(plot_width=300, plot_height=300)

    hist, edges = np.histogram(df_gof[metric].fillna(0), bins=bins)
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], alpha=0.4)

    p.title.text = metric.upper()

    return p


def plot_metrics(df_gof, bins=100):
    gp = bokeh.layouts.gridplot(
        [
            [
                plot_metric_histogram(df_gof, metric='l2_norm', bins=bins),
                plot_metric_histogram(df_gof, metric='earth_mover', bins=bins),
                plot_metric_histogram(df_gof, metric='entropy', bins=bins),
            ],
            [
                plot_metric_histogram(df_gof, metric='kld', bins=bins),
                plot_metric_histogram(df_gof, metric='slope', bins=bins),
                plot_metric_histogram(df_gof, metric='chi2_stats', bins=bins),
            ],
        ]
    )

    bokeh.plotting.show(gp)


def plot_slopes(
    x_corpus: VectorizedCorpus, most_deviating: pd.DataFrame, metric: str, plot_height=300, plot_width=300
) -> Dict:
    def generate_slopes(x_corpus: VectorizedCorpus, most_deviating: pd.DataFrame, metric: str):

        min_year = x_corpus.documents.year.min()
        max_year = x_corpus.documents.year.max()
        xs = np.arange(min_year, max_year + 1, 1)
        token_ids = [x_corpus.token2id[token] for token in most_deviating[metric + '_token']]
        data = collections.defaultdict(list)
        # plyfit of all columns: kx_m = np.polyfit(x=xs, y=x_corpus.data[:,token_ids], deg=1)
        for token_id in token_ids:
            ys = x_corpus.data[:, token_id]
            data["token_id"].append(token_id)
            data["token"].append(x_corpus.id2token[token_id])
            _, k, p, lx, ly = fit_ordinary_least_square(ys, xs)
            data['k'].append(k)
            data['p'].append(p)
            data['xs'].append(np.array(lx))
            data['ys'].append(np.array(ly))
        return data

    data = generate_slopes(x_corpus, most_deviating, metric)

    source = bokeh.models.ColumnDataSource(data)

    color_mapper = bokeh.models.LinearColorMapper(palette='Magma256', low=min(data['k']), high=max(data['k']))

    p = bokeh.plotting.figure(plot_height=plot_height, plot_width=plot_width, tools='pan,wheel_zoom,box_zoom,reset')
    p.multi_line(
        xs='xs',
        ys='ys',
        line_width=1,
        line_color={'field': 'k', 'transform': color_mapper},
        line_alpha=0.6,
        hover_line_alpha=1.0,
        source=source,
    )  # , legend="token"

    p.add_tools(
        bokeh.models.HoverTool(
            show_arrow=False,
            line_policy='next',
            tooltips=[('Token', '@token'), ('Slope', '@k{1.1111}')],  # , ('P-value', '@p{1.1111}')]
        )
    )

    bokeh.plotting.show(p)

    # return p
