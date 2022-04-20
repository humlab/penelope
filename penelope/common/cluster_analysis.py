import abc
import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import linkage

import penelope.common.goodness_of_fit as gof
from penelope.corpus import VectorizedCorpus

try:
    import sklearn
    import sklearn.cluster
except ImportError:
    ...


@dataclass
class KMeansResult:
    centroids: np.ndarray  # A ‘k’ by ‘N’ array of centroids found at the last iteration of k-means.
    labels: np.ndarray  # label[i] is the code or index of the centroid the ith observation is closest to.


class CorpusClusters(abc.ABC):
    def __init__(self, corpus: VectorizedCorpus, tokens: List[str]):
        self._token_clusters: pd.DataFrame = None
        self.corpus: VectorizedCorpus = corpus
        self.tokens: List[str] = tokens
        self.cluster_labels = []

    @property
    def n_clusters(self) -> int:
        return len(self.cluster_labels)

    @property
    def token_clusters(self) -> pd.DataFrame:
        return self._token_clusters

    @token_clusters.setter
    def token_clusters(self, value: pd.DataFrame):
        self._token_clusters = value
        self.cluster_labels = (
            [] if self.token_clusters is None else sorted(self.token_clusters.cluster.unique().tolist())
        )

    def cluster_token_ids(self, label: str) -> List[int]:
        return self.token_clusters[self.token_clusters.cluster == label].index.tolist()

    def clusters_token_ids(self) -> Iterable[Tuple[str, List[int]]]:

        for label in self.cluster_labels:
            yield label, self.cluster_token_ids(label)

    def cluster_means(self) -> np.ndarray:

        cluster_means: np.ndarray = np.array(
            [self.corpus.data[:, token_ids].mean(axis=1) for _, token_ids in self.clusters_token_ids()]
        )

        return cluster_means

    def cluster_medians(self) -> np.ndarray:
        try:
            data = self.corpus.data.todense()
            cluster_medians: np.ndarray = np.array(
                [np.median(data[:, token_ids], axis=1) for _, token_ids in self.clusters_token_ids()]
            )

            return cluster_medians
        except np.AxisError:
            return None

    @property
    @abc.abstractproperty
    def threshold(self) -> float:
        ...

    @threshold.setter
    @abc.abstractproperty
    def threshold(self, value: float):
        ...


class HCACorpusClusters(CorpusClusters):
    def __init__(self, corpus: VectorizedCorpus, tokens: List[str], linkage_matrix, threshold: float = 0.5):

        super().__init__(corpus, tokens)

        self._threshold = 0.0

        self.key = 'hca'
        self.token2id: Dict[str, int] = corpus.token2id
        self.linkage_matrix = linkage_matrix
        self.cluster_distances = self._compile_cluster_distances(linkage_matrix)
        self.cluster2tokens: Dict[str, Set[str]] = None
        self.threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value
        self.cluster2tokens: Dict[str, Set[str]] = self._reduce_to_threshold(self._threshold)
        self.token_clusters = self._compile_token_clusters(self.cluster2tokens)
        self.cluster2tokens = self.cluster2tokens

    def _compile_cluster_distances(self, linkage_matrix) -> pd.DataFrame:
        """Returns a data frame with cluster distances"""
        N = len(self.tokens)

        df = pd.DataFrame(data=linkage_matrix, columns=['a_id', 'b_id', 'distance', 'n_obs']).astype(
            {'a_id': np.int64, 'b_id': np.int64, 'n_obs': np.int64}
        )

        df['a_cluster'] = df.a_id.apply(lambda i: self.tokens[i] if i < N else '#{}#'.format(i))
        df['b_cluster'] = df.b_id.apply(lambda i: self.tokens[i] if i < N else '#{}#'.format(i))
        df['cluster'] = ['#{}#'.format(N + i) for i in df.index]

        df = df[['a_cluster', 'b_cluster', 'distance', 'cluster']]  # , 'a_id', 'b_id', 'n_obs']]
        return df

    def _reduce_to_threshold(self, threshold: float) -> Dict[str, Set[str]]:
        """Reduces clusters to given threshold"""
        cluster2tokens = {x: set([x]) for x in self.tokens}
        for _, r in self.cluster_distances.iterrows():
            if r['distance'] > threshold:
                break
            cluster2tokens[r['cluster']] = set(cluster2tokens[r['a_cluster']]) | set(cluster2tokens[r['b_cluster']])
            del cluster2tokens[r['a_cluster']]
            del cluster2tokens[r['b_cluster']]

        return cluster2tokens

    def _compile_token_clusters(self, clusters: Dict[str, Set[str]]) -> pd.DataFrame:
        cluster_lists = [[(x, y, i) for x, y in itertools.product([k], clusters[k])] for i, k in enumerate(clusters)]
        df: pd.DataFrame = pd.DataFrame(
            data=[x for ws in cluster_lists for x in ws], columns=["cluster_name", "token", "cluster"]
        )
        df['token_id'] = df.token.apply(lambda w: self.token2id[w])
        return df.set_index('token_id')


class KMeansCorpusClusters(CorpusClusters):
    def __init__(self, corpus: VectorizedCorpus, tokens: List[str], kmean_result: KMeansResult):

        super().__init__(corpus, tokens)

        self.key = 'k_means'
        self.token2id = corpus.token2id
        # self.kmean_result = kmean_result
        self.token2cluster = self.create_token2cluster(kmean_result.labels)
        self.token_clusters = self.creater_cluster_data_frame(self.token2cluster)
        self.centroids = kmean_result.centroids

    def create_token2cluster(self, labels):
        token2cluster = {self.tokens[i]: label for i, label in enumerate(labels)}
        return token2cluster

    def creater_cluster_data_frame(self, token2cluster):

        df = pd.DataFrame({'token': list(token2cluster.keys()), 'cluster': list(token2cluster.values())})

        df['token_id'] = df.token.apply(lambda w: self.token2id[w])

        return df.set_index('token_id')

    @property
    def threshold(self) -> float:
        raise ValueError("kmeans: threshold not supported")

    @threshold.setter
    def threshold(self, _: float):
        raise ValueError("kmeans: threshold not supported")


def compute_kmeans(corpus: VectorizedCorpus, tokens: List[str] = None, n_clusters: int = 8, **kwargs):
    """Computes KMeans clusters using `sklearn.cluster.KMeans`(https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)"""
    data: scipy.sparse.spmatrix = corpus.data if tokens is None else corpus.data[:, corpus.token_indices(tokens)]

    km = sklearn.cluster.KMeans(n_clusters=n_clusters, **kwargs).fit(data.T)

    return KMeansCorpusClusters(corpus, tokens, KMeansResult(centroids=km.cluster_centers_, labels=km.labels_))


def compute_kmeans2(corpus: VectorizedCorpus, tokens: List[str] = None, n_clusters: int = 8, **kwargs):
    """Computes KMeans clusters using `scipy.cluster.vq.kmeans2` (https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans2.html"""
    data: scipy.sparse.spmatrix = corpus.data if tokens is None else corpus.data[:, corpus.token_indices(tokens)]
    data = data.T.todense()
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float64)
    centroids, labels = scipy.cluster.vq.kmeans2(data, n_clusters, **kwargs)

    return KMeansCorpusClusters(corpus, tokens, KMeansResult(centroids=centroids, labels=labels))


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


def compute_hca(
    corpus: VectorizedCorpus, tokens: List[str], linkage_method: str = 'ward', linkage_metric: str = 'euclidean'
) -> HCACorpusClusters:
    """Computes HCA clusters using `scipy.cluster.hierarchy.linkage` (https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html"""
    data = corpus.data if tokens is None else corpus.data[:, corpus.token_indices(tokens)]

    linkage_matrix = linkage(data.T.todense(), method=linkage_method, metric=linkage_metric)
    """ from documentation

        A (n-1) by 4 matrix Z is returned. At the i-th iteration, clusters with token_ids Z[i, 0] and Z[i, 1] are combined to form cluster n + i.
        A cluster with an index less than n corresponds to one of the original observations.
        The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2].
        The fourth value Z[i, 3] represents the number of original observations in the newly formed cluster.

    """

    return HCACorpusClusters(corpus, tokens, linkage_matrix)


def smooth_array(xs, ys, smoothers):
    _xs = xs
    _ys = ys.copy()
    for smoother in smoothers or []:
        _xs, _ys = smoother(_xs, _ys)
    return _xs, _ys


def smooth_matrix(xs, ys_m, smoothers):

    return zip(*[smooth_array(xs, ys_m[:, i], smoothers) for i in range(0, ys_m.shape[1])])


def get_top_tokens_by_metric(
    *, metric: str, n_metric_top: int, corpus: VectorizedCorpus, df_gof: pd.DataFrame
) -> Tuple[List[int], List[str]]:
    """Computes most deviating tokens by metric"""
    df_top = gof.get_most_deviating_words(df_gof, metric, n_top=n_metric_top, ascending=False, abs_value=False)
    tokens = df_top[metric + '_token'].tolist()
    indices = [corpus.token2id[w] for w in tokens]
    return indices, tokens


def compute_clusters(
    *,
    method_key: str,
    n_clusters: int,
    metric: str,
    n_metric_top: int,
    corpus: VectorizedCorpus,
    df_gof: pd.DataFrame,
) -> CorpusClusters:
    """Computes cluster analysis data from specified parameters on `corpus` using distance metrics in df_god"""
    _, tokens = get_top_tokens_by_metric(metric=metric, n_metric_top=n_metric_top, corpus=corpus, df_gof=df_gof)

    if method_key == 'k_means++':
        cluster_data = compute_kmeans(corpus, tokens, n_clusters, init='k-means++')
    elif method_key == 'k_means':
        cluster_data = compute_kmeans(corpus, tokens, n_clusters, init='random')
    elif method_key == 'k_means2':
        cluster_data = compute_kmeans2(corpus, tokens, n_clusters)
    else:
        cluster_data = compute_hca(corpus, tokens, linkage_method='ward', linkage_metric='euclidean')

    return cluster_data
