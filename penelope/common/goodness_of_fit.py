import collections
from dataclasses import dataclass
from typing import Dict, List

import bokeh
import numpy as np
import pandas as pd
import scipy
from pandas.core.frame import DataFrame
from penelope.corpus import VectorizedCorpus
from penelope.utility import chunks
from tqdm.auto import tqdm

from .distance_metrics import (
    earth_mover_distance,
    entropy,
    fit_ordinary_least_square,
    fit_polynomial,
    gof_by_l2_norm,
    gof_chisquare_to_uniform,
    kullback_leibler_divergence_to_uniform,
)


class GoodnessOfFitComputeError(ValueError):
    ...


@dataclass
class GofData:

    goodness_of_fit: pd.DataFrame = None
    most_deviating_overview: pd.DataFrame = None
    most_deviating: pd.DataFrame = None

    @staticmethod
    def compute(corpus: VectorizedCorpus, n_count: int) -> "GofData":

        goodness_of_fit = compute_goddness_of_fits_to_uniform(corpus, None, verbose=True, metrics=['l2_norm', 'slope'])
        most_deviating_overview = compile_most_deviating_words(goodness_of_fit, n_count=n_count)
        most_deviating = get_most_deviating_words(
            goodness_of_fit, 'l2_norm', n_count=n_count, ascending=False, abs_value=True
        )

        gof_data: GofData = GofData(
            goodness_of_fit=goodness_of_fit,
            most_deviating=most_deviating,
            most_deviating_overview=most_deviating_overview,
        )

        return gof_data


def get_gof_by_l2_norms(dtm: scipy.sparse.spmatrix) -> pd.DataFrame:
    df_gof = pd.DataFrame(
        {
            'l2_norm': gof_by_l2_norm(dtm, axis=0),
        }
    )
    return df_gof


def get_gof_by_polynomial(dtm: scipy.sparse.spmatrix, x_offset: float = 0.0) -> DataFrame:
    try:
        if isinstance(dtm, scipy.sparse.spmatrix):
            dtm = dtm.todense()
        xs = [x + x_offset for x in range(0, dtm.shape[0])]
        fitted_values = fit_polynomial(xs=xs, ys=dtm, deg=1)
    except:  # pylint: disable=bare-except
        fitted_values = (np.nan, np.nan)

    return pd.DataFrame(
        {'slope': fitted_values[1], 'intercept': fitted_values[0]}, index=range(0, dtm.shape[1]), dtype=np.float64
    )


def get_gof_chisquare_to_uniform(dtm: scipy.sparse.spmatrix) -> pd.DataFrame:
    try:
        chi2_stats, chi2_p = list(
            zip(*[gof_chisquare_to_uniform(dtm.getcol(i).A.ravel()) for i in range(0, dtm.shape[1])])
        )
    except:  # pylint: disable=bare-except
        chi2_stats, chi2_p = np.nan, np.nan
    return pd.DataFrame({'chi2_stats': chi2_stats, 'chi2_p': chi2_p}, index=range(0, dtm.shape[1]), dtype=np.float64)


def get_earth_mover_distance(dtm: scipy.sparse.spmatrix) -> pd.DataFrame:
    try:
        emd = [earth_mover_distance(dtm.getcol(i).A.ravel()) for i in range(0, dtm.shape[1])]
    except:  # pylint: disable=bare-except
        emd = np.nan
    return pd.DataFrame({'earth_mover': emd}, index=range(0, dtm.shape[1]), dtype=np.float64)


def get_entropy_to_uniform(dtm: scipy.sparse.spmatrix) -> pd.DataFrame:
    try:
        e = [entropy(dtm.getcol(i).A.ravel()) for i in range(0, dtm.shape[1])]
    except:  # pylint: disable=bare-except
        e = np.nan
    return pd.DataFrame({'entropy': e}, index=range(0, dtm.shape[1]), dtype=np.float64)


def get_kullback_leibler_divergence_to_uniform(dtm: scipy.sparse.spmatrix) -> pd.DataFrame:
    try:
        kld = [kullback_leibler_divergence_to_uniform(dtm.getcol(i).A.ravel()) for i in range(0, dtm.shape[1])]
    except:  # pylint: disable=bare-except
        kld = np.nan
    return pd.DataFrame({'kld': kld}, index=range(0, dtm.shape[1]), dtype=np.float64)


def get_skew(dtm: scipy.sparse.spmatrix) -> pd.DataFrame:
    try:
        if not isinstance(dtm, scipy.sparse.spmatrix):
            raise GoodnessOfFitComputeError("get_skew expects a sparse matrixs")
        skew = [scipy.stats.skew(dtm.getcol(i).A.ravel()) for i in range(0, dtm.shape[1])]
    except:  # pylint: disable=bare-except
        skew = np.nan
    return pd.DataFrame({'skew': skew}, index=range(0, dtm.shape[1]), dtype=np.float64)


def get_basic_statistics(dtm: scipy.sparse.spmatrix) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'min': [dtm[:, i].min() for i in range(0, dtm.shape[1])],
            'max': [dtm[:, i].max() for i in range(0, dtm.shape[1])],
            'mean': [dtm[:, i].mean() for i in range(0, dtm.shape[1])],
            # 'var': [np.var(dtm[:, i]) for i in range(0, dtm.shape[1])],
        },
        index=range(0, dtm.shape[1]),
        dtype=np.float64,
    )


METRIC_FUNCTIONS = {
    'l2_norm': get_gof_by_l2_norms,
    "slope": get_gof_by_polynomial,
    "chi2_stats": get_gof_chisquare_to_uniform,
    'stats': get_basic_statistics,
    'earth_mover': get_earth_mover_distance,
    'entropy': get_entropy_to_uniform,
    'kld': get_kullback_leibler_divergence_to_uniform,
    'skew': get_skew,
}


def compute_goddness_of_fits_to_uniform(
    corpus: VectorizedCorpus, n_top_count: int = None, verbose=False, metrics: List[str] = None
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
    metrics = metrics or list(METRIC_FUNCTIONS.keys())

    dtm = corpus.data  # .todense()

    if dtm.shape[0] <= 1:
        raise GoodnessOfFitComputeError("Unable to compute GoF (to few bags)")

    if dtm.shape[1] == 0:
        raise GoodnessOfFitComputeError("Unable to compute GoF (no terms supplied)")

    df_gof = pd.DataFrame(
        {
            'token': [corpus.id2token[i] for i in range(0, dtm.shape[1])],
            'word_count': [corpus.term_frequency[i] for i in range(0, dtm.shape[1])],
        }
    )

    metrics_iter = tqdm(metrics, desc="Computing metrics", position=0, leave=False) if verbose else metrics
    for metric in metrics_iter:
        if metric in METRIC_FUNCTIONS:
            if hasattr(metrics_iter, 'set_description'):
                metrics_iter.set_description(metric)
            df_gof = df_gof.join(METRIC_FUNCTIONS[metric](dtm))

    df_gof.sort_values(['l2_norm'], ascending=False, inplace=True)

    if (n_top_count or 0) > 0:
        df_gof = df_gof.nlargest(n_top_count, columns=["word_count"])

    if not verbose:

        if 'stats' in metrics:
            metrics.remove('stats')

        df_gof = df_gof[
            [
                "token",
                "word_count",
            ]
            + metrics
        ]
    return df_gof


def get_most_deviating_words(
    df_gof: pd.DataFrame, metric: str, n_count: int = 500, ascending: bool = False, abs_value: bool = False
):
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

    computed_metrics = list(set(df.columns) - set(["token", "word_count"]))
    xf = df[["token", "word_count"]]
    for metric in computed_metrics:
        xf = xf.join(get_most_deviating_words(df, metric, n_count, abs_value=True))
    return xf


def plot_metric_histogram(df_gof, metric='l2_norm', bins=100):

    p = bokeh.plotting.figure(plot_width=300, plot_height=300)

    hist, edges = np.histogram(df_gof[metric].fillna(0), bins=bins)
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], alpha=0.4)

    p.title.text = metric.upper()

    return p


def plot_metrics(df_gof, bins=100):
    plots = []
    for metric in METRIC_FUNCTIONS:
        if metric in df_gof.columns:
            plots.append(plot_metric_histogram(df_gof, metric=metric, bins=bins))

    plots = chunks(list(plots), 3)

    gp = bokeh.layouts.gridplot(plots)

    bokeh.plotting.show(gp)


def generate_slopes(x_corpus: VectorizedCorpus, most_deviating: pd.DataFrame, metric: str):

    min_year = x_corpus.document_index.year.min()
    max_year = x_corpus.document_index.year.max()
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


def plot_slopes(
    x_corpus: VectorizedCorpus, most_deviating: pd.DataFrame, metric: str, plot_height: int = 300, plot_width: int = 300
) -> Dict:

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
