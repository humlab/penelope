import types
from unittest import mock

import numpy as np
import pandas as pd
import penelope.common.goodness_of_fit as gof
import penelope.notebook.cluster_analysis.cluster_analysis_gui as cluster_analysis_gui
from penelope.corpus import dtm


def dummy_patch(*_, **__):
    ...


def monkey_patch(*_, **__):
    ...


@mock.patch('IPython.display.display', dummy_patch)
def test_create_gui():

    corpus = mock.Mock(spec=dtm.VectorizedCorpus)
    df_gof = mock.Mock(spec=pd.DataFrame)
    container = cluster_analysis_gui.display_gui(corpus, df_gof)

    assert container is not None


def create_vectorized_corpus():
    bag_term_matrix = np.array([[2, 1, 4, 1], [2, 2, 3, 0], [2, 3, 2, 0], [2, 4, 1, 1], [2, 0, 1, 1]])
    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    document_index = pd.DataFrame({'year': [2013, 2013, 2014, 2014, 2014]})
    corpus = dtm.VectorizedCorpus(bag_term_matrix, token2id, document_index)
    return corpus


@mock.patch('IPython.display.display', dummy_patch)
@mock.patch('bokeh.plotting.show', dummy_patch)
@mock.patch('penelope.notebook.cluster_analysis.cluster_plot.create_cluster_plot', monkey_patch)
@mock.patch('penelope.notebook.cluster_analysis.cluster_plot.render_cluster_plot', monkey_patch)
@mock.patch('penelope.notebook.cluster_analysis.cluster_plot.create_cluster_boxplot', monkey_patch)
@mock.patch('penelope.notebook.cluster_analysis.cluster_plot.render_cluster_boxplot', monkey_patch)
@mock.patch('penelope.notebook.cluster_analysis.cluster_plot.plot_clusters_count', monkey_patch)
@mock.patch('penelope.notebook.cluster_analysis.cluster_plot.render_pandas_frame', monkey_patch)
@mock.patch('penelope.notebook.cluster_analysis.cluster_plot.create_clusters_mean_plot', monkey_patch)
@mock.patch('penelope.notebook.cluster_analysis.cluster_plot.render_clusters_mean_plot', monkey_patch)
@mock.patch('penelope.notebook.cluster_analysis.cluster_plot.plot_dendogram', monkey_patch)
def test_setup_gui():

    corpus = create_vectorized_corpus()
    df_gof = gof.compute_goddness_of_fits_to_uniform(corpus=corpus)

    state = cluster_analysis_gui.GUI_State(corpus_clusters=None, corpus=corpus, df_gof=df_gof)
    gui = cluster_analysis_gui.ClusterAnalysisGUI(state=state, display_trends=dummy_patch)

    # gui.plot_cluster = monkey_patch
    # gui.plot_clusters = monkey_patch
    gui.plot_clusters_mean = monkey_patch
    gui.plot_clusters_count = monkey_patch

    gui.setup()

    gui.lock(True)
    assert gui.forward.disabled

    gui.lock(False)
    assert not gui.forward.disabled

    assert gui.method_key.value == 'k_means2'
    assert gui.metric.value == 'l2_norm'

    layout = gui.layout()
    assert layout is not None

    gui.compute_clicked({})

    assert len(gui.cluster_index.options) > 0
    assert gui.current_index == 0
    gui.step_cluster(types.SimpleNamespace(description=">>"))
    assert gui.current_index == 1
    gui.step_cluster(types.SimpleNamespace(description="<<"))
    assert gui.current_index == 0

    for output_type in ['count', 'dendrogram', 'table']:
        gui.clusters_output_type.value = output_type
        gui.compute_clicked({})

    gui.threshold_range_changed({})
