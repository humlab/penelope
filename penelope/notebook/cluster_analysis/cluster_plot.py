import itertools
from typing import Iterable

import bokeh
import bokeh.models as bm
import bokeh.plotting as bp
import holoviews as hv
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import penelope.common.curve_fit as cf
from penelope.common import distance_metrics
from penelope.corpus import VectorizedCorpus
from penelope.utility import nth
from scipy.cluster.hierarchy import dendrogram, linkage


def default_palette(index: int) -> Iterable[str]:

    return nth(itertools.cycle(bokeh.palettes.Category20[20]), index)


def noop(x=None, p=None, max=None):  # pylint: disable=redefined-builtin, unused-argument
    pass


# pylint: disable=too-many-locals
def create_cluster_plot(
    x_corpus: VectorizedCorpus, token_clusters: pd.DataFrame, n_cluster: int, tick=noop, **kwargs
) -> bp.Figure:

    # palette = itertools.cycle(bokeh.palettes.Category20[20])
    assert n_cluster <= token_clusters.cluster.max()

    xs = np.arange(x_corpus.document_index.year.min(), x_corpus.document_index.year.max() + 1, 1)
    token_ids = list(token_clusters[token_clusters.cluster == n_cluster].index)
    word_distributions = x_corpus.data.todense()[:, token_ids]

    tick(1, max=len(token_ids))

    title: str = kwargs.get('title', 'Cluster #{}'.format(n_cluster))

    p: bp.Figure = bp.figure(
        title=title,
        plot_width=kwargs.get('plot_width', 900),
        plot_height=kwargs.get('plot_height', 600),
        output_backend="webgl",
    )

    p.yaxis.axis_label = 'Frequency'
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.ticker = xs
    p.xaxis.major_label_orientation = 3.14 / 2
    p.y_range.start = 0

    xsr = np.repeat(xs, word_distributions.shape[1])
    ysr = word_distributions.ravel()

    p.scatter(xs=xsr, ys=ysr, size=3, color='green', alpha=0.1, marker='square', legend_label='actual')

    p.line(xs=xsr, ys=ysr, line_width=1.0, color='green', alpha=0.1, legend_label='actual')

    tick(1)
    _, _, _, lx, ly = distance_metrics.fit_ordinary_least_square_ravel(word_distributions, xs)
    p.line(xs=lx, ys=ly, line_width=0.6, color='black', alpha=0.8, legend_label='trend')

    xsp, ysp = cf.fit_curve_ravel(cf.polynomial3, xs, word_distributions)
    p.line(xs=xsp, ys=ysp, line_width=0.5, color='blue', alpha=0.5, legend_label='poly3')

    if hasattr(token_clusters, 'centroids'):

        ys_cluster_center = token_clusters.centroids[n_cluster, :]
        p.line(xs=xs, ys=ys_cluster_center, line_width=2.0, color='black', legend_label='centroid')

        xs_spline, ys_spline = cf.pchip_spline(xs, ys_cluster_center)
        p.line(xs=xs_spline, ys=ys_spline, line_width=2.0, color='red', legend_label='centroid (pchip)')

    else:
        ys_mean = word_distributions.mean(axis=1)
        ys_median = np.median(word_distributions, axis=1)

        xs_spline, ys_spline = cf.pchip_spline(xs, ys_mean)
        p.line(xs=xs_spline, ys=ys_spline, line_width=2.0, color='red', legend_label='mean (pchip)')

        xs_spline, ys_spline = cf.pchip_spline(xs, ys_median)
        p.line(xs=xs_spline, ys=ys_spline, line_width=2.0, color='blue', legend_label='median (pchip)')

    tick(2)

    return p


def render_cluster_plot(figure: bp.Figure):
    bp.show(figure)


def create_cluster_boxplot(
    x_corpus: VectorizedCorpus, token_clusters: pd.DataFrame, n_cluster: int, color: str
) -> hv.opts:

    xs = np.arange(x_corpus.document_index.year.min(), x_corpus.document_index.year.max() + 1, 1)

    token_ids = list(token_clusters[token_clusters.cluster == n_cluster].index)

    Y = x_corpus.data[:, token_ids]
    xsr = np.repeat(xs, Y.shape[1])
    ysr = Y.ravel()

    data = pd.DataFrame(data={'year': xsr, 'frequency': ysr})

    violin: hv.BoxWhisker = hv.BoxWhisker(data, ('year', 'Year'), ('frequency', 'Frequency'))

    violin_opts = {
        'height': 600,
        'width': 900,
        'box_fill_color': color,
        'xrotation': 90,
        # 'violin_width': 0.8
    }
    return violin.opts(**violin_opts)


def render_cluster_boxplot(p: hv.opts):
    p = hv.render(p)
    bp.show(p)
    return p


def plot_clusters_count(source: bm.ColumnDataSource):

    figure_opts = dict(plot_width=500, plot_height=600, title="Cluster token count")

    hover_opts = dict(tooltips='@legend: @count words', show_arrow=False, line_policy='next')

    bar_opts = dict(
        legend_field='legend',
        fill_color='color',
        fill_alpha=0.4,
        hover_fill_alpha=1.0,
        hover_fill_color='color',
        line_color='color',
        hover_line_color='color',
        line_alpha=1.0,
        hover_line_alpha=1.0,
        height=0.75,
    )

    p = bp.figure(tools=[bm.HoverTool(**hover_opts), bm.TapTool()], **figure_opts)

    # y_range=source.data['clusters'],
    p.yaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.x_range.start = 0

    _ = p.hbar(source=source, y='cluster', right='count', **bar_opts)

    return p


def create_clusters_mean_plot(source: bm.ColumnDataSource, filter_source: dict = None) -> bm.Box:

    figure_opts = dict(plot_width=600, plot_height=620, title="Cluster mean trends (pchip spline)")
    hover_opts = dict(tooltips=[('Cluster', '@legend')], show_arrow=False, line_policy='next')

    line_opts = dict(
        legend_field='legend',
        line_color='color',
        line_width=5,
        line_alpha=0.4,
        hover_line_color='color',
        hover_line_alpha=1.0,
    )

    p: bp.Figure = bp.figure(tools=[bm.HoverTool(**hover_opts), bm.TapTool()], **figure_opts)

    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    # p.xaxis.ticker = list(range(1945, 1990))
    p.y_range.start = 0

    _ = p.multi_line(source=source, xs='xs', ys='ys', **line_opts)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    if filter_source is not None:

        callback: bm.CustomJS = _create_multiline_multiselect_callback(source)

        multi_select: bm.MultiSelect = bm.MultiSelect(
            title='Show/hide',
            options=filter_source['options'],
            value=filter_source['values'],
            size=min(len(filter_source['options']), 30),
        )

        multi_select.js_on_change('value', callback)

        p: bm.Box = bokeh.layouts.row(p, multi_select)

    return p


def render_clusters_mean_plot(figure: bm.Box):
    bp.show(figure)


def _create_multiline_multiselect_callback(source: bm.ColumnDataSource) -> bm.CustomJS:

    full_source = bm.ColumnDataSource(source.data)

    callback: bm.CustomJS = bm.CustomJS(
        args=dict(source=source, full_source=full_source),
        code="""
        const indices = cb_obj.value.map(x => parseInt(x));
        let items = ['xs', 'ys', 'color', 'legend'];
        for (const item of items) {
            let full_item = full_source.data[item];
            source.data[item].length = 0;
            for (var i of indices) {
                source.data[item].push(full_item[i]);
            }
        }
        source.change.emit()
        """,
    )
    return callback


def render_dendogram(linkage_matrix):
    dendrogram(linkage(linkage_matrix, 'ward'))


def plot_dendogram(linkage_matrix, labels):

    plt.figure(figsize=(16, 40))

    dendrogram(
        linkage_matrix,
        truncate_mode="level",
        color_threshold=1.8,
        show_leaf_counts=True,
        no_labels=False,
        orientation="right",
        labels=labels,
        leaf_rotation=0,  # rotates the x axis labels
        leaf_font_size=12,  # font size for the x axis labels
    )
    plt.show()


def render_pandas_frame(df: pd.DataFrame):
    display.display(df)
