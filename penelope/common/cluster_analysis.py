import itertools
import types

import numpy as np
import pandas as pd
import scipy
import sklearn
import sklearn.cluster
from scipy.cluster.hierarchy import linkage


class CorpusClusters:
    def __init__(self, corpus, tokens):
        self._token_clusters = None
        self.corpus = corpus
        self.tokens = tokens
        self.cluster_labels = []

    @property
    def n_clusters(self):
        return len(self.cluster_labels)

    @property
    def token_clusters(self):
        return self._token_clusters

    @token_clusters.setter
    def token_clusters(self, value):
        self._token_clusters = value
        self.cluster_labels = (
            [] if self.token_clusters is None else sorted(self.token_clusters.cluster.unique().tolist())
        )

    def cluster_token_ids(self, n_cluster):
        return self.token_clusters[self.token_clusters.cluster == n_cluster].index.tolist()

    def clusters_token_ids(self):

        for n_cluster in self.cluster_labels:
            yield n_cluster, self.cluster_token_ids(n_cluster)

    def cluster_means(self):

        cluster_means = np.array(
            [self.corpus.data[:, token_ids].mean(axis=1) for _, token_ids in self.clusters_token_ids()]
        )

        return cluster_means

    def cluster_medians(self):

        cluster_medians = np.array(
            [np.median(self.corpus.data[:, token_ids], axis=1) for _, token_ids in self.clusters_token_ids()]
        )

        return cluster_medians


class HCACorpusClusters(CorpusClusters):
    def __init__(self, corpus, tokens, linkage_matrix, threshold=0.5):

        super().__init__(corpus, tokens)

        self.key = 'hca'

        self.token2id = corpus.token2id
        self.linkage_matrix = linkage_matrix
        self.cluster_distances = self._compile_cluster_distances(linkage_matrix)
        self.threshold = threshold
        self.cluster2tokens = None
        self.set_threshold(threshold=self.threshold)

    def set_threshold(self, threshold):

        self.cluster2tokens = self._reduce_to_threshold(threshold)
        self.token_clusters = self._compile_token_clusters(self.cluster2tokens)
        self.cluster2tokens = self.cluster2tokens
        self.threshold = threshold

    def _compile_cluster_distances(self, linkage_matrix):

        N = len(self.tokens)

        df = pd.DataFrame(data=linkage_matrix, columns=['a_id', 'b_id', 'distance', 'n_obs']).astype(
            {'a_id': np.int64, 'b_id': np.int64, 'n_obs': np.int64}
        )

        df['a_cluster'] = df.a_id.apply(lambda i: self.tokens[i] if i < N else '#{}#'.format(i))
        df['b_cluster'] = df.b_id.apply(lambda i: self.tokens[i] if i < N else '#{}#'.format(i))
        df['cluster'] = ['#{}#'.format(N + i) for i in df.index]

        df = df[['a_cluster', 'b_cluster', 'distance', 'cluster']]  # , 'a_id', 'b_id', 'n_obs']]
        return df

    def _reduce_to_threshold(self, threshold):

        cluster2tokens = {x: set([x]) for x in self.tokens}
        for _, r in self.cluster_distances.iterrows():
            if r['distance'] > threshold:
                break
            cluster2tokens[r['cluster']] = set(cluster2tokens[r['a_cluster']]) | set(cluster2tokens[r['b_cluster']])
            del cluster2tokens[r['a_cluster']]
            del cluster2tokens[r['b_cluster']]

        return cluster2tokens

    def _compile_token_clusters(self, clusters):
        cluster_lists = [[(x, y, i) for x, y in itertools.product([k], clusters[k])] for i, k in enumerate(clusters)]
        df = pd.DataFrame(data=[x for ws in cluster_lists for x in ws], columns=["cluster_name", "token", "cluster"])
        df['token_id'] = df.token.apply(lambda w: self.token2id[w])
        return df.set_index('token_id')


class KMeansCorpusClusters(CorpusClusters):
    def __init__(self, corpus, tokens, compute_result):

        super().__init__(corpus, tokens)

        self.key = 'k_means'
        self.token2id = corpus.token2id
        self.compute_result = compute_result
        self.token2cluster = self._compile_token2cluster_map(compute_result.labels_)
        self.token_clusters = self._compile_token_clusters(self.token2cluster)
        self.centroids = compute_result.cluster_centers_

    def _compile_token2cluster_map(self, labels):
        token2cluster = {self.tokens[i]: label for i, label in enumerate(labels)}
        return token2cluster

    def _compile_token_clusters(self, token2cluster):

        df = pd.DataFrame({'token': list(token2cluster.keys()), 'cluster': list(token2cluster.values())})

        df['token_id'] = df.token.apply(lambda w: self.token2id[w])

        return df.set_index('token_id')


def compute_kmeans(x_corpus, tokens=None, n_clusters=8, **kwargs):

    data = x_corpus.data if tokens is None else x_corpus.data[:, x_corpus.token_indices(tokens)]

    compute_result = sklearn.cluster.KMeans(n_clusters=n_clusters, **kwargs).fit(data.T)

    return KMeansCorpusClusters(x_corpus, tokens, compute_result)


def compute_kmeans2(x_corpus, tokens=None, n_clusters=8, **kwargs):

    data = x_corpus.data if tokens is None else x_corpus.data[:, x_corpus.token_indices(tokens)]

    centroids, labels = scipy.cluster.vq.kmeans2(data.T, n_clusters, **kwargs)

    return KMeansCorpusClusters(x_corpus, tokens, types.SimpleNamespace(cluster_centers_=centroids, labels_=labels))


LINKAGE_METHODS = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']

LINKAGE_METRICS = {
    'braycurtis': 'Bray-Curtis distance.',
    'canberra': 'Canberra distance.',
    'chebyshev': 'Chebyshev distance.',
    'cityblock': 'Manhattan distance.',
    'correlation': 'Correlation distance.',
    'cosine': 'Cosine distance.',
    'euclidean': 'Euclidean distance.',
    'jensenshannon': 'Jensen-Shannon distance.',
    'mahalanobis': 'Mahalanobis distance.',
    'minkowski': 'Minkowski distance.',
    'seuclidean': 'Normalized Euclidean distance.',
    'sqeuclidean': 'Squared Euclidean distance.',
}


def compute_hca(x_corpus, tokens, linkage_method='ward', linkage_metric='euclidean'):

    data = x_corpus.data if tokens is None else x_corpus.data[:, x_corpus.token_indices(tokens)]

    linkage_matrix = linkage(data.T, method=linkage_method, metric=linkage_metric)
    """ from documentation

        A (n-1) by 4 matrix Z is returned. At the i-th iteration, clusters with token_ids Z[i, 0] and Z[i, 1] are combined to form cluster n + i.
        A cluster with an index less than n corresponds to one of the original observations.
        The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2].
        The fourth value Z[i, 3] represents the number of original observations in the newly formed cluster.

    """

    return HCACorpusClusters(x_corpus, tokens, linkage_matrix)
