import bokeh
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import penelope.common.curve_fit as cf
from bokeh.models import HoverTool, TapTool
from penelope.common import distance_metrics
from penelope.corpus import VectorizedCorpus
from scipy.cluster.hierarchy import dendrogram


def noop(x=None, p=None, max=None):  # pylint: disable=redefined-builtin, unused-argument
    pass


# pylint: disable=too-many-locals
def plot_cluster(x_corpus: VectorizedCorpus, token_clusters, n_cluster, tick=noop, **kwargs):

    # palette = itertools.cycle(bokeh.palettes.Category20[20])
    assert n_cluster <= token_clusters.cluster.max()

    xs = np.arange(x_corpus.documents.year.min(), x_corpus.documents.year.max() + 1, 1)
    token_ids = list(token_clusters[token_clusters.cluster == n_cluster].index)
    word_distributions = x_corpus.data.todense()[:, token_ids]

    tick(1, max=len(token_ids))

    title = kwargs.get('title', 'Cluster #{}'.format(n_cluster))

    p = bokeh.plotting.figure(
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


def plot_cluster_boxplot(x_corpus: VectorizedCorpus, token_clusters, n_cluster, color):

    xs = np.arange(x_corpus.documents.year.min(), x_corpus.documents.year.max() + 1, 1)

    token_ids = list(token_clusters[token_clusters.cluster == n_cluster].index)

    Y = x_corpus.data[:, token_ids]
    xsr = np.repeat(xs, Y.shape[1])
    ysr = Y.ravel()

    data = pd.DataFrame(data={'year': xsr, 'frequency': ysr})

    kind = hv.BoxWhisker  # hv.Violin
    violin = kind(data, ('year', 'Year'), ('frequency', 'Frequency'))

    violin_opts = {
        'height': 600,
        'width': 900,
        'box_fill_color': color,
        'xrotation': 90,
        # 'violin_width': 0.8
    }
    return violin.opts(**violin_opts)


def plot_clusters_count(source):

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

    p = bokeh.plotting.figure(tools=[HoverTool(**hover_opts), TapTool()], **figure_opts)

    # y_range=source.data['clusters'],
    p.yaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.x_range.start = 0

    _ = p.hbar(source=source, y='cluster', right='count', **bar_opts)

    return p


def plot_clusters_mean(source, filter_source=None):

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

    p = bokeh.plotting.figure(tools=[HoverTool(**hover_opts), TapTool()], **figure_opts)

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

        callback = create_multiline_multiselect_callback(source)

        multi_select = bokeh.models.MultiSelect(
            title='Show/hide',
            options=filter_source['options'],
            value=filter_source['values'],
            size=min(len(filter_source['options']), 30),
        )

        multi_select.js_on_change('value', callback)

        p = bokeh.layouts.row(p, multi_select)

    return p


def create_multiline_multiselect_callback(source):

    full_source = bokeh.models.ColumnDataSource(source.data)

    callback = bokeh.models.CustomJS(
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


# from ipywidgets import interact, interactive, fixed, interact_manual
# import ipywidgets as widgets

# def plot_seaborn_clustermap(x_corpus, token_clusters, n_cluster=0):

#     token_ids          = list(token_clusters[token_clusters.cluster==n_cluster].index)
#     tokens             = [ x_corpus.id2token[token_id] for token_id in token_ids ]
#     word_distributions = x_corpus.data[:,token_ids].T
#     xs                 = np.arange(x_corpus.documents.year.min(), x_corpus.documents.year.max() + 1, 1)
#     df                 = pd.DataFrame(data=word_distributions[:,:], index=tokens, columns=[str(x) for x in xs])

#     sns.clustermap(df, metric="correlation", method="single", cmap="Blues", standard_scale=1) #, row_colors=row_colors)

# token_clusters = cluster_analysis_gui.DEBUG_CONTAINER['data'].token_clusters

# interact(
#     plot_seaborn_clustermap,
#     x_corpus=fixed(n_corpus),
#     token_clusters=fixed(token_clusters),
#     n_cluster=token_clusters.cluster.unique().tolist()
# )

# #plot_seaborn_clustermap(n_corpus, cluster_analysis_gui.CURRENT_CLUSTER.clusters.token_clusters)
