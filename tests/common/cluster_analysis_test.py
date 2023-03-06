import numpy as np
import pandas as pd
import pytest

import penelope.common.goodness_of_fit as gof
from penelope.common.cluster_analysis import CorpusClusters, compute_clusters
from penelope.corpus import VectorizedCorpus


def create_vectorized_corpus():
    bag_term_matrix = np.array([[2, 1, 4, 1], [2, 2, 3, 0], [2, 3, 2, 0], [2, 4, 1, 1], [2, 0, 1, 1]])
    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    document_index = pd.DataFrame({'year': [2013, 2013, 2014, 2014, 2014]})
    corpus = VectorizedCorpus(bag_term_matrix, token2id=token2id, document_index=document_index)
    return corpus


@pytest.mark.parametrize(
    'method_key,metric',
    [
        ('k_means++', 'l2_norm'),
        ('k_means', 'l2_norm'),
        ('k_means2', 'l2_norm'),
        ('hca', 'l2_norm'),
    ],
)
def test_setup_gui(
    method_key: str,
    metric: str,
):
    n_cluster_count: int = 3
    n_metric_top: int = 4

    corpus = create_vectorized_corpus()
    df_gof = gof.compute_goddness_of_fits_to_uniform(corpus=corpus)

    corpus_clusters: CorpusClusters = compute_clusters(
        method_key=method_key,
        n_clusters=n_cluster_count,
        metric=metric,
        n_metric_top=n_metric_top,
        corpus=corpus,
        df_gof=df_gof,
    )
    assert corpus_clusters is not None
    assert corpus_clusters.n_clusters > 0

    assert len([x for x in corpus_clusters.clusters_token_ids()]) > 0
    assert corpus_clusters.cluster_means() is not None
    assert corpus_clusters.cluster_medians() is not None

    if method_key == 'hca':
        corpus_clusters.threshold = 0.9
        assert corpus_clusters.threshold == 0.9
