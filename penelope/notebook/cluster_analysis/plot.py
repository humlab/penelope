import itertools

import bokeh
import penelope.common.curve_fit as cf
from penelope.common.cluster_analysis import smooth_matrix
from penelope.corpus import DocumentIndexHelper

from . import cluster_plot


def create_filter_source(token_counts):

    # cluster_info = token_counts['token'].sort_values().to_dict()
    cluster_info = token_counts['token'].to_dict()

    cluster_options = [(str(n), 'Cluster {}, {} types'.format(n, wc)) for (n, wc) in cluster_info.items()]
    cluster_values = [n for (n, _) in cluster_options]

    return dict(options=cluster_options, values=cluster_values)


class ClustersMeanPlot:
    def __init__(self):

        self.source = None
        self.figure = None
        self.smoothers = [cf.rolling_average_smoother('nearest', 3), cf.pchip_spline]

    def update(self, x_corpus, ys_matrix, token_counts=None, smoothers=None) -> "ClustersMeanPlot":

        filter_source = create_filter_source(token_counts)

        colors = itertools.cycle(bokeh.palettes.Category20[20])

        smoothers = smoothers or self.smoothers
        xs = DocumentIndexHelper.xs_years(x_corpus.document_index)

        smoothers = None  # []

        ml_xs, ml_ys = smooth_matrix(xs, ys_matrix, smoothers)
        ml_colors = list(itertools.islice(colors, ys_matrix.shape[1]))
        ml_legends = ['cluster {}'.format(i) for i in range(0, ys_matrix.shape[1])]

        self.source = bokeh.models.ColumnDataSource(dict(xs=ml_xs, ys=ml_ys, color=ml_colors, legend=ml_legends))

        self.figure = cluster_plot.create_clusters_mean_plot(source=self.source, filter_source=filter_source)

        cluster_plot.render_clusters_mean_plot(self.figure)

        return self

    def plot(self) -> "ClustersMeanPlot":
        bokeh.plotting.show(self.figure)
        return self


# pylint: disable=too-many-locals, too-many-statements
class ClustersCountPlot:
    def __init__(self):

        self.figure = None
        # self.source = bokeh.models.ColumnDataSource(dict(cluster=[1,2,3], count=[1,2,3]))

        #     self.figure = cluster_plot.plot_clusters_count(source=self.source)
        #     self.handle = bokeh.plotting.show(self.figure, notebook_handle=True)

    def update(self, token_counts) -> "ClustersCountPlot":

        colors = itertools.cycle(bokeh.palettes.Category20[20])

        source = dict(
            cluster=[x for x in token_counts.index],  # [ str(x) for x in token_counts.index ],
            count=[x for x in token_counts.token],
            legend=['cluster {}'.format(i) for i in token_counts.index],
            color=[next(colors) for _ in token_counts.index],
        )
        self.figure = cluster_plot.plot_clusters_count(source=source)
        # self.source.data = source
        # bokeh.io.push_notebook(self.handle)
        return self

    def plot(self) -> "ClustersCountPlot":
        bokeh.plotting.show(self.figure)
        return self
