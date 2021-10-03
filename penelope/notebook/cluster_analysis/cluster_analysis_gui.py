import warnings
from dataclasses import dataclass
from typing import Callable, List

import IPython.display as display
import pandas as pd
from ipywidgets import HTML, Button, Dropdown, FloatSlider, HBox, IntSlider, Layout, Output, VBox
from penelope.common.cluster_analysis import CorpusClusters, compute_clusters
from penelope.corpus import VectorizedCorpus
from penelope.utility import clamp, get_logger

from ..word_trends import trends_with_picks_gui as word_trend_plot_gui
from . import cluster_plot
from .plot import ClustersCountPlot, ClustersMeanPlot

warnings.filterwarnings("ignore", category=FutureWarning)

logger = get_logger()

DEBUG_CONTAINER = {'data': None}

# pylint: disable=too-many-instance-attributes

CLUSTER_OUTPUT_TYPES = [('Scatter', 'scatter'), ('Boxplot', 'boxplot')]
CLUSTERS_OUTPUT_TYPES = [('Bar', 'count'), ('Dendogram', 'dendrogram'), ('Table', 'table')]
METRICS_LIST = [('L2-norm', 'l2_norm'), ('EMD', 'emd'), ('KLD', 'kld')]
METHODS_LIST = [('K-means++', 'k_means++'), ('K-means', 'k_means'), ('K-means/scipy', 'k_means2'), ('HCA', 'hca')]
N_METRIC_TOP_WORDS = [10, 100, 250, 500, 1000, 2000, 5000, 10000, 20000]


@dataclass
class GUI_State:

    corpus_clusters: CorpusClusters = None
    corpus: VectorizedCorpus = None
    df_gof: pd.DataFrame = None

    def get_cluster_tokens(self, cluster: str) -> List[str]:
        token_clusters = self.corpus_clusters.token_clusters
        tokens = token_clusters[token_clusters.cluster == cluster].token.tolist()
        return tokens


class ClusterAnalysisGUI:
    def __init__(self, state: GUI_State, display_callback: Callable):

        self.state = state
        self.display_trends = display_callback
        self.n_cluster_count: IntSlider = IntSlider(
            description='#Cluster', min=1, max=200, step=1, value=20, bar_style='info', continuous_update=False
        )
        self.method_key: Dropdown = Dropdown(
            description='Method', options=METHODS_LIST, value='k_means2', layout=Layout(width='200px')
        )
        self.metric: Dropdown = Dropdown(
            description='Metric', options=METRICS_LIST, value='l2_norm', layout=Layout(width='200px')
        )
        self.n_metric_top: Dropdown = Dropdown(
            description='Words', options=N_METRIC_TOP_WORDS, value=5000, layout=Layout(width='200px')
        )
        self.clusters_output_type: Dropdown = Dropdown(
            description='Output', options=CLUSTERS_OUTPUT_TYPES, value='count', layout=Layout(width='200px')
        )
        self.compute = Button(description='Compute', button_style='Success', layout=Layout(width='100px'))
        self.clusters_count_output: Output = Output()
        self.clusters_mean_output: Output = Output()
        self.cluster_output_type: Dropdown = Dropdown(
            description='Output', options=CLUSTER_OUTPUT_TYPES, value='boxplot', layout=Layout(width='200px')
        )
        self.threshold: FloatSlider = FloatSlider(
            description='Threshold', min=0.0, max=1.0, step=0.01, value=0.50, bar_style='info', continuous_update=False
        )
        self.cluster_index: Dropdown = Dropdown(
            description='Cluster', value=None, options=[], bar_style='info', disabled=True, layout=Layout(width='200px')
        )
        self.back: Button = Button(
            description="<<", button_style='Success', layout=Layout(width='40px', color='green'), disabled=True
        )
        self.forward: Button = Button(
            description=">>", button_style='Success', layout=Layout(width='40px', color='green'), disabled=True
        )
        self.cluster_output: Output = Output()
        self.cluster_words_output: Output = Output()

    def lock(self, value: bool):
        self.compute.disabled = value
        self.forward.disabled = value
        self.back.disabled = value
        self.cluster_index.disabled = value
        if value:
            self.cluster_index.unobserve(self._plot_cluster, 'value')
        else:
            self.cluster_index.observe(self._plot_cluster, 'value')

    def layout(self) -> VBox:

        return VBox(
            [
                HBox(
                    [
                        VBox(
                            [
                                HBox(
                                    [
                                        HTML("Select method, number of clusters and press compute."),
                                        VBox([self.method_key, self.metric, self.n_metric_top]),
                                        VBox(
                                            [
                                                self.n_cluster_count,
                                                HBox([self.clusters_output_type, self.compute]),
                                            ],
                                            layout=Layout(align_items='flex-end'),
                                        ),
                                    ]
                                ),
                                HTML("<h2> Clusters overview</h2>"),
                                HBox([self.clusters_count_output, self.clusters_mean_output]),
                            ]
                        )
                    ]
                ),
                VBox(
                    [
                        VBox(
                            [
                                HTML("<h1>Browse cluster</h2>"),
                                HBox([self.cluster_output_type, self.threshold]),
                                HBox([self.cluster_index, self.back, self.forward]),
                                self.cluster_output,
                            ]
                        ),
                    ]
                ),
                HTML("<h2>Explore words in cluster</h2>"),
                self.cluster_words_output,
            ]
        )

    def setup(self) -> "ClusterAnalysisGUI":

        self.forward.on_click(self.step_cluster)
        self.back.on_click(self.step_cluster)
        self.compute.on_click(self.compute_clicked)
        self.method_key.observe(self.set_method, 'value')
        self.cluster_index.observe(self._plot_cluster, 'value')
        self.cluster_output_type.observe(self._plot_cluster, 'value')
        self.threshold.observe(self.threshold_range_changed, 'value')
        self.set_method()
        return self

    @property
    def current_index(self) -> int:
        if len(self.cluster_index.options) == 0:
            return None
        if self.cluster_index.value is None:
            return None
        return list(self.cluster_index.options).index(self.cluster_index.value)

    def step_cluster(self, b):
        step = -1 if b.description == "<<" else 1
        next_index = clamp(self.current_index + step, 0, len(self.cluster_index.options) - 1)
        self.cluster_index.value = self.cluster_index.options[next_index]

    def set_method(self, *_):

        wth, wnc = self.threshold, self.n_cluster_count
        if self.method_key.value == 'hca':
            wnc.min, wnc.max, wnc.value = 0, 0, 0
            wth.min, wth.max, wth.value = 0.0, 1.0, 0.5
            wnc.disabled, wth.disabled = True, False
        else:
            wnc.max, wnc.value, wnc.min = 250, 8, 2
            wnc.disabled, wth.disabled = False, True

    def _plot_cluster(self, *_):
        self.plot_cluster()

    def _plot_clusters(self, *_):
        self.plot_clusters()

    def plot_words(self):

        tokens = self.state.get_cluster_tokens(self.cluster_index.value)
        if len(tokens) == 0:
            return

        self.cluster_words_output.clear_output()
        with self.cluster_words_output:
            self.display_trends(self.state.corpus, tokens, n_columns=3)

        # plot_cluster()

    def plot_cluster(self):

        self.cluster_output.clear_output()

        if self.state.corpus_clusters is None:
            return

        token_clusters = self.state.corpus_clusters.token_clusters

        with self.cluster_output:

            out_table: Output = Output()
            out_chart: Output = Output()

            display.display(HBox([out_table, out_chart]))

            with out_chart:
                if self.cluster_output_type.value == "scatter":
                    p = cluster_plot.create_cluster_plot(self.state.corpus, token_clusters, self.cluster_index.value)
                    cluster_plot.render_cluster_plot(p)

                if self.cluster_output_type.value == "boxplot":
                    color = cluster_plot.default_palette(self.cluster_index.value)
                    p = cluster_plot.create_cluster_boxplot(
                        self.state.corpus, token_clusters, self.cluster_index.value, color=color
                    )
                    cluster_plot.render_cluster_boxplot(p)

            with out_table:
                df = token_clusters[token_clusters.cluster == self.cluster_index.value]
                cluster_plot.render_pandas_frame(df)

        self.plot_words()

    def plot_clusters(self):

        output_type = self.clusters_output_type.value
        token_clusters = self.state.corpus_clusters.token_clusters
        token_counts = token_clusters.groupby('cluster').count()

        if output_type == 'count':
            self.clusters_count_output.clear_output()
            with self.clusters_count_output:
                self.plot_clusters_count(token_counts)

        if output_type == 'dendrogram' and self.state.corpus_clusters.key == 'hca':
            cluster_plot.render_dendogram(self.state.corpus_clusters.linkage_matrix)

        if output_type == 'table':
            cluster_plot.render_pandas_frame(token_clusters)

        if output_type == 'heatmap':
            print('Heatmap: not implemented')

        self.clusters_mean_output.clear_output()
        with self.clusters_mean_output:
            self.plot_clusters_mean(token_counts)

    def plot_clusters_mean(self, token_counts):
        ClustersMeanPlot().update(
            self.state.corpus,
            ys_matrix=self.state.corpus_clusters.cluster_means().T,
            token_counts=token_counts,
        ).plot()

    def plot_clusters_count(self, token_counts):
        ClustersCountPlot().update(token_counts=token_counts).plot()

    def threshold_range_changed(self, *_):

        if self.threshold.disabled is True:
            return

        self.state.corpus_clusters.threshold = self.threshold.value

        self.cluster_index.unobserve(self._plot_cluster, 'value')
        self._plot_clusters()

        self.cluster_index.options = self.state.corpus_clusters.cluster_labels
        self.cluster_index.value = self.cluster_index.options[0] if len(self.cluster_index.options) > 0 else None

        self.cluster_index.observe(self._plot_cluster, 'value')

        self._plot_cluster()

    def compute_clicked(self, *_):

        self.lock(True)

        self.cluster_output.clear_output()

        with self.cluster_output:
            print("Working, please wait...")

        try:
            self.cluster_index.value = None
            self.cluster_index.options = []

            self.state.corpus_clusters = compute_clusters(
                method_key=self.method_key.value,
                n_clusters=self.n_cluster_count.value,
                metric=self.metric.value,
                n_metric_top=self.n_metric_top.value,
                corpus=self.state.corpus,
                df_gof=self.state.df_gof,
            )
            self.cluster_index.options = self.state.corpus_clusters.cluster_labels

            with self.clusters_count_output:
                self.plot_clusters()

            self.plot_cluster()

            if len(self.cluster_index.options) > 0:
                self.cluster_index.value = self.cluster_index.options[0]

        except Exception as ex:
            with self.cluster_output:
                logger.exception(ex)
            raise

        self.lock(False)


def display_trends(corpus: VectorizedCorpus, tokens: List[str], n_columns: int = 3):
    gui = word_trend_plot_gui.TrendsWithPickTokensGUI.create(corpus, tokens, n_columns=n_columns)
    display.display(gui.layout())


def display_gui(corpus: VectorizedCorpus, df_gof: pd.DataFrame):

    state = GUI_State(corpus_clusters=None, corpus=corpus, df_gof=df_gof)

    DEBUG_CONTAINER['data'] = state

    gui = ClusterAnalysisGUI(state=state, display_callback=display_trends).setup()
    layout = gui.layout()
    display.display(layout)

    return state
