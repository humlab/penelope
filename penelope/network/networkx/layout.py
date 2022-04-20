from penelope.utility import extend_single

from ..interface import LayoutAlgorithm
from .networkx_api import nx
from .utility import get_bipartite_node_set


def nx_kamada_kawai_layout(G, **kwargs):  # pylint: disable=unused-argument
    args = dict(weight='weight', scale=1.0)
    layout = nx.kamada_kawai_layout(G, **args)
    return layout, None


def nx_spring_layout(G: nx.Graph, **kwargs):
    k = kwargs.get('K', 0.1)
    args = dict(weight='weight', scale=1.0, k=k)
    args = extend_single(args, kwargs, 'iterations')
    layout = nx.spring_layout(G, **args)
    return layout, None


def nx_shell_layout(G, **kwargs):  # pylint: disable=unused-argument
    if not nx.is_bipartite(G):
        raise Exception("NX: Shell layout only applicable on bipartite graphs")
    nodes, other_nodes = get_bipartite_node_set(G, bipartite=0)
    layout = nx.shell_layout(G, nlist=[nodes, other_nodes])
    return layout, None


def nx_spectral_layout(G, **kwargs):  # pylint: disable=unused-argument
    layout = nx.spectral_layout(G)
    return layout, None


def nx_circular_layout(G, **kwargs):  # pylint: disable=unused-argument
    layout = nx.circular_layout(G)
    return layout, None


layout_setups = [
    LayoutAlgorithm(key='nx_spring_layout', package='nx', name='nx_spring', layout_network=nx_spring_layout),
    LayoutAlgorithm(key='nx_spectral_layout', package='nx', name='nx_spectral', layout_network=nx_spectral_layout),
    LayoutAlgorithm(key='nx_circular_layout', package='nx', name='nx_circular', layout_network=nx_circular_layout),
    LayoutAlgorithm(key='nx_shell_layout', package='nx', name='nx_shell', layout_network=nx_shell_layout),
    LayoutAlgorithm(
        key='nx_kamada_kawai_layout', package='nx', name='nx_kamada_kawai', layout_network=nx_kamada_kawai_layout
    ),
]
