import networkx as nx

from penelope.network.metrics import compute_centrality, compute_partition, partition_colors

# def compute_partition(network):
#     partition = community_louvain.best_partition(network)  # pylint: disable=no-member
#     _, nodes_community = zip(*sorted(partition.items()))
#     return nodes_community


def nx_to_python_louvain(partition):
    return {n: c for c, ns in enumerate(partition) for n in ns}


def python_louvain_to_nx(partition):
    return {value: [n for n, c in partition.items() if c == value] for value in set(partition.values())}


def compute_partition(network):
    partition = nx.community.louvain_communities(network)
    partition_map: dict = {n: c for c, ns in enumerate(partition) for n in ns}
    partition_by_index = [partition_map[n] for n in sorted(partition_map.keys())]
    return partition_by_index


def test_metrics_compute_partition():
    # Test graph In this graph, with two distinct communities {1, 2, 3} and {4, 5, 6}.
    # 1 -- 2 -- 3
    # |         |
    # 4 -- 5 -- 6

    G = nx.Graph()

    G.add_nodes_from([1, 2, 3, 4, 5, 6])
    G.add_edges_from([(1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (3, 6)])

    expected = [0, 1, 1, 0, 2, 2]

    nx_partition = nx.community.louvain_communities(G, seed=42)

    assert nx_partition == [{1, 4}, {2, 3}, {5, 6}]

    nx_partition_map: dict = {n: c for c, ns in enumerate(nx_partition) for n in ns}
    assert nx_partition_map == {1: 0, 4: 0, 2: 1, 3: 1, 5: 2, 6: 2}

    nx_partition_by_index = [nx_partition_map[n] for n in sorted(nx_partition_map.keys())]

    assert nx_partition_by_index == expected

    data = compute_centrality(G)
    assert data is not None

    assert partition_colors(nx_partition_by_index) == ['#e41a1c', '#377eb8', '#377eb8', '#e41a1c', '#4daf4a', '#4daf4a']
