from numbers import Number
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
from penelope.utility import clamp_values, extend, list_of_dicts_to_dict_of_lists

Attributes = Dict[str, Any]

Node = Any
Edge = Tuple[Node, Node, Attributes]
Weight = float
Point = Tuple[Number, Number]
EdgeEndPoints = Tuple[Number, Number, Number, Number]
EdgesLayout = Tuple[Node, Node, Weight, EdgeEndPoints]
NodesLayout = Dict[Node, Point]


def layout_edges(network: nx.Graph, layout: NodesLayout, weight: str = 'weight') -> EdgesLayout:
    """Extracts edges´layout data

    Args:
        network (nx.Graph): The network.
        layout (NodesLayout): Network layout
        weight (str, optional): Name if weight attribute. Defaults to 'weight'.

    Returns:
        List[Any, Any, Number, Sequence[Number,Number,Number,Number]]: List of edges as (N, M, w, [Nx, Ny, Mx, My])
    """
    data = [
        (u, v, d[weight], [layout[u][0], layout[v][0]], [layout[u][1], layout[v][1]])
        for u, v, d in network.edges(data=True)
    ]

    return zip(*data)


def pandas_to_edges(df: pd.DataFrame, source: str = 'source', target: str = 'target', **edge_attributes) -> List[Edge]:
    """Transform a dataframe's edge data into nx style representation i.e. as a list of (source, target, attributes) triplets
    Any other column are stored as attributes in an attr dictionary

    Args:
        df (pd.DataFrame):
        source (str, optional): Source column. Defaults to 'source'.
        target (str, optional): Target column. Defaults to 'target'.

    Returns:
        List[Tuple[Node, Node, Attributes]]:  Edges represented in nx style as a list of (source, target, attributes) triplets
            i.e [ ('source-node', 'target-node', { 'attr_1': value, ...., 'attr-n': value })]

    """

    attr_fields = list(edge_attributes.values())
    attr_names = {v: k for k, v in edge_attributes.items()}

    edges = zip(
        df[source].values,
        df[target].values,
        df[attr_fields].apply(lambda x: {attr_names[k]: v for k, v in x.to_dict().items()}, axis=1),
    )
    return list(edges)


def df_to_nx(
    df: pd.DataFrame, source: str = 'source', target: str = 'target', bipartite: bool = False, **edge_attributes
) -> nx.Graph:
    """Creates a new networkx graph from values in a dataframe.

    Args:
        df (pd.DataFrame): The source data frame.
        source (str, optional): Source node column. Defaults to 'source'.
        target (str, optional): Target node column. Defaults to 'target'.
        bipartite (bool, optional): If specified then a bipartite graph is created. Defaults to False.

    Returns:
        nx.Graph: [description]

    Example:
        df = pd.DataFrame({'A': [1,2,3,4,5], 'B': [6,7,8,9,10], 'W': [1,2,3,3,3]})
        df_to_nx(df, source_field='A', target_field='B', weight='W')

    """
    G = nx.Graph()

    if bipartite:

        source_nodes = set(df[source].values)
        target_nodes = set(df[target].values)

        assert len(source_nodes.intersection(target_nodes)) == 0, "Bipartite graph cannot have overlapping node names!"

        G.add_nodes_from(source_nodes, bipartite=0)
        G.add_nodes_from(target_nodes, bipartite=1)
    else:
        G.add_nodes_from(list(set(df[source]).union(set(df[target]))))

    edges = pandas_to_edges(df, source=source, target=target, **edge_attributes)

    G.add_edges_from(edges)

    return G


def create_network(
    df: pd.DataFrame, source_field: str = 'source', target_field: str = 'target', weight: str = 'weight'
) -> nx.Graph:
    """Creates a network from data in a pandas data frame"""
    G = nx.Graph()
    nodes = list(set(list(df[source_field].values) + list(df[target_field].values)))
    edges = [(x, y, {'weight': z}) for x, y, z in [tuple(x) for x in df[[source_field, target_field, weight]].values]]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def create_bipartite_network(
    df: pd.DataFrame, source_field: str = 'source', target_field: str = 'target', weight: float = 'weight'
) -> nx.Graph():
    """Create as bipartite networkx graph from columns in a pandas dataframe"""
    G = nx.Graph()
    G.add_nodes_from(set(df[source_field].values), bipartite=0)
    G.add_nodes_from(set(df[target_field].values), bipartite=1)
    edges = list(zip(df[source_field].values, df[target_field].values, df[weight].apply(lambda x: dict(weight=x))))
    G.add_edges_from(edges)
    return G


def get_sub_network(G: nx.Graph, attribute: str = 'weight', threshold: float = 0.0) -> nx.Graph:
    """Creates a subgraph of g of all edges having a attribute value equal to or above threshold.
    Threshold is a value in percent where max attribute value = 100%. Defaults to 0.0.
    """
    max_weight = max(1.0, max(nx.get_edge_attributes(G, attribute).values()))
    filter_edges = [(u, v) for u, v, d in G.edges(data=True) if d[attribute] >= (threshold * max_weight)]
    tng = G.edge_subgraph(filter_edges)
    return tng


def get_positioned_nodes(network: nx.Graph, layout: NodesLayout, nodes: List[str] = None) -> Dict[str, List]:
    """Returns nodes layout data as a dict  of lists.

    Args:
        network (nx.Graph): The network
        layout (NodesLayout): A dictionary with node coordinates.
        nodes (List[str], optional): Subset of nodes to return.. Defaults to None.

    Returns:
        Dict[List]: Positioned nodes (xs, ys, nodes, node_id, ...attributes) and any additional found attributes
    """
    layout_items = layout.items() if nodes is None else [x for x in layout.items() if x[0] in nodes]

    nodes, nodes_coordinates = zip(*sorted(layout_items))
    xs, ys = list(zip(*nodes_coordinates))

    list_of_attribs = [network.nodes[k] for k in nodes]
    attrib_lists = dict(zip(list_of_attribs[0], zip(*[d.values() for d in list_of_attribs])))
    attrib_lists.update(dict(x=xs, y=ys, name=nodes, node_id=nodes))
    dict_of_lists = {k: list(v) for k, v in attrib_lists.items()}

    return dict_of_lists


def get_positioned_edges(network: nx.Graph, layout: NodesLayout, sort_attr: str = None) -> List[Dict]:
    """Extracts network edge attributes and endpoint coordinates, plus midpont coordinate

    Args:
        network (nx.Graph): [description]
        layout (List[Dict]): Dictionary of node coordinate pairs i.e. { N: (N, [x,y]) }
        sort_attr ([type], optional): Sort attribute. Defaults to None.

    Returns:
        List[Dict]: Positioned edges and attributes
         i.e. {
             source:  source node,
             target:  target-node,
             xs:      [x1, x2],
             ys:      [y1, y2],
             m_x:     (x1 + x2) / 2,
             y_x:     (y1 + y2) / 2,
             attr-1:  value of attr-1
             ...
             attr-n:  value of attr-n
        }
        x1, y1     source node's coordinate
        x2, y2     target node's coordinate
        m_x, m_y   midpoint coordinare
    """

    list_of_dicts = [
        extend(
            dict(
                source=u,
                target=v,
                xs=[layout[u][0], layout[v][0]],
                ys=[layout[u][1], layout[v][1]],
                m_x=[(layout[u][0] + layout[v][0]) / 2.0],
                m_y=[(layout[u][1] + layout[v][1]) / 2.0],
            ),
            d,
        )
        for u, v, d in network.edges(data=True)
    ]

    if sort_attr is not None:
        list_of_dicts.sort(key=lambda x: x[sort_attr])

    return list_of_dicts


def get_positioned_edges_as_dict(network: nx.Graph, layout: NodesLayout, sort_attr=None) -> Dict[str, List]:
    """Returns positioned edges and all associated attributes.
    Is simply a reformat of result from get_positioned_edges
    Args:
        network (nx.Graph): The networkx graph.
        layout (List[Dict]): Dictionary of node coordinate pairs i.e. { N: (N, [x,y]) }
        sort_attr ([type], optional): Sort attribute. Defaults to None.

    Returns:
        Dict[List]: Positioned edges as a dict of edge-attribute lists
         i.e. {
             source:  [list of source nodes],
             target:  [list of target nodes],
             xs:      [list of [x1, x2]],
             ys:      [list of [y1, y2]],
             m_x:     [list of (x2-x1)],
             y_x:     [list of (y2-y1)],
             weight:  [list of weights],
             ...attrs lists of any additional attributes found
        }

        m_x, m_y = (x_target + x_source) / 2, (y_target + y_source) / 2
            computed by midpoint formula

    Parameters
    ----------
    network : nx.Graph
        The networkx graph.

    layout : dict of node + point pairs i.e. (node, [x,y])
        A dictionary that contains coordinates for all nodes.

    Returns
    -------


    """
    list_of_dicts = get_positioned_edges(network, layout, sort_attr)

    dict_of_tuples = list_of_dicts_to_dict_of_lists(list_of_dicts)
    dict_of_lists = {k: list(v) for k, v in dict_of_tuples.items()}  # convert tuples to lists

    return dict_of_lists


def get_positioned_nodes_as_dict(
    G: nx.Graph, layout: NodesLayout, node_size: str, node_size_range: Optional[Tuple[Number, Number]]
) -> dict:

    nodes = get_positioned_nodes(G, layout)

    if node_size in nodes.keys() and node_size_range is not None:
        nodes['clamped_size'] = clamp_values(nodes[node_size], node_size_range)
        node_size = 'clamped_size'

    label_y_offset = 'y_offset' if node_size in nodes.keys() else node_size + 8
    if label_y_offset == 'y_offset':
        nodes['y_offset'] = [y + r for (y, r) in zip(nodes['y'], [r / 2.0 + 8 for r in nodes[node_size]])]

    nodes = {k: list(nodes[k]) for k in nodes}

    return nodes


def get_bipartite_node_set(network: nx.Graph, bipartite: int = 0) -> Tuple[List[Any], List[Any]]:
    """Extracts nodes from a bipartites graph"""
    nodes = set(n for n, d in network.nodes(data=True) if d['bipartite'] == bipartite)
    others = set(network) - nodes
    return list(nodes), list(others)
