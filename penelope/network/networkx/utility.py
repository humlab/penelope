import networkx as nx
from penelope.utility import clamp_values, extend, list_of_dicts_to_dict_of_lists


def get_edge_layout_data(network, layout, weight='weight'):

    data = [
        (u, v, d[weight], [layout[u][0], layout[v][0]], [layout[u][1], layout[v][1]])
        for u, v, d in network.edges(data=True)
    ]

    return zip(*data)


def df_to_nx_edge_format(data, source_index=0, target_index=1):
    """Transform a dataframe's edge data into nx style i.e. as a list of (source, target, attributes) triplets

    The source and target nodes are assumed to be the first to columns.
    Any other column are stored as attributes in an attr dictionary

    Parameters
    ----------
    data : DataFrame
        A pandas dataframe that contains edges i.e. source/target/weight columns.
            df.columns = ['source', 'target', 'attr_1', 'attr_2', 'attr_n']

    source_index, target_index: int
        Source and target column index.

    Returns
    -------
        Edges represented in nx style as a list of (source, target, attributes) triplets
            i.e [ ('source-node', 'target-node', { 'attr_1': value, ...., 'attr-n': value })]

    """
    return [
        (x[source_index], x[target_index], {y: x[j] for j, y in enumerate(data.columns)}) for i, x in data.iterrows()
    ]


def df_to_nx_edges_list(df, source='source', target='target', **edge_attributes):
    """More generic implementation"""
    attr_fields = list(edge_attributes.values())
    attr_names = {v: k for k, v in edge_attributes.items()}

    edges = zip(
        df[source].values,
        df[target].values,
        df[attr_fields].apply(lambda x: {attr_names[k]: v for k, v in x.to_dict().items()}, axis=1),
    )
    return list(edges)


def df_to_nx(df, source='source', target='target', bipartite=False, **edge_attributes):
    """Creates a new networkx graph from values in a dataframe.

    Parameters
    ----------
    df : DataFrame
        A pandas dataframe that contains edges i.e. source/target/weight columns.

    source : str
        Name of column that contains source nodes.

    target : str
        Name of column that contains target nodes.

    edge_attributes : str
        Name of edge attribute columns.

    Returns
    -------
        A networkx Graph.

    Example:
    -------
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

    edges = df_to_nx_edges_list(df, source=source, target=target, **edge_attributes)

    G.add_edges_from(edges)

    return G


create_nx_graph = df_to_nx


def get_subgraph(g, attribute='weight', threshold=0.0):
    """Creates a subgraph of g of all edges having a attribute value equal to or above threshold.

    Parameters
    ----------
    network : nx.Graph
        The networkx graph.

    threshold : float
        Threshold in percent where max attribute value = 100%

    Returns
    -------
        A networkx sub-graph

    """
    max_weight = max(1.0, max(nx.get_edge_attributes(g, attribute).values()))
    filter_edges = [(u, v) for u, v, d in g.edges(data=True) if d[attribute] >= (threshold * max_weight)]
    tng = g.edge_subgraph(filter_edges)
    return tng


get_sub_network = get_subgraph


def get_positioned_nodes(network, layout, nodes=None):
    """Returns nodes assigned position from given layout.

    Parameters
    ----------
    network : nx.Graph
        The networkx graph.

    layout : dict of node + point pairs i.e. (node, [x,y])
        A dictionary that contains coordinates for all nodes.

    nodes : optional list of str
        Subset of nodes to return.

    Returns : dict of lists
    -------
        Positioned nodes (xs, ys, nodes, node_id, ...attributes) and any additional found attributes

    """
    layout_items = layout.items() if nodes is None else [x for x in layout.items() if x[0] in nodes]

    nodes, nodes_coordinates = zip(*sorted(layout_items))
    xs, ys = list(zip(*nodes_coordinates))

    list_of_attribs = [network.nodes[k] for k in nodes]

    attrib_lists = dict(zip(list_of_attribs[0], zip(*[d.values() for d in list_of_attribs])))

    attrib_lists.update(dict(x=xs, y=ys, name=nodes, node_id=nodes))

    dict_of_lists = {k: list(v) for k, v in attrib_lists.items()}

    return dict_of_lists


def get_positioned_edges(network, layout, sort_attr=None):
    """Extracts network edge attributes and assigns coordinates to endpoints, and computes midpont coordinate

    Parameters
    ----------
    network : nx.Graph
        The networkx graph.

    layout : dict of node + point pairs i.e. (node, [x,y])
        A dictionary that contains coordinates for all nodes.

    Returns
    -------
        Return list of dicts
         i.e. {
             source:  source node,
             target:  target-node,
             xs:      [x1, x2],
             ys:      [y1, y2],           # Y-coordinate
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


def get_positioned_edges2(network, layout, sort_attr=None):
    """Returns positioned edges as and all associated attributes.

    Is simply a reformat of result from get_layout_edges_attributes

    Parameters
    ----------
    network : nx.Graph
        The networkx graph.

    layout : dict of node + point pairs i.e. (node, [x,y])
        A dictionary that contains coordinates for all nodes.

    Returns
    -------

        Positioned edges as a dict of edge-attribute lists
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

    """
    list_of_dicts = get_positioned_edges(network, layout, sort_attr)

    dict_of_tuples = list_of_dicts_to_dict_of_lists(list_of_dicts)
    dict_of_lists = {k: list(v) for k, v in dict_of_tuples.items()}  # convert tuples to lists

    return dict_of_lists


def get_positioned_nodes_as_dict(G, layout, node_size, node_size_range):

    nodes = get_positioned_nodes(G, layout)

    if node_size in nodes.keys() and node_size_range is not None:
        nodes['clamped_size'] = clamp_values(nodes[node_size], node_size_range)
        node_size = 'clamped_size'

    label_y_offset = 'y_offset' if node_size in nodes.keys() else node_size + 8
    if label_y_offset == 'y_offset':
        nodes['y_offset'] = [y + r for (y, r) in zip(nodes['y'], [r / 2.0 + 8 for r in nodes[node_size]])]

    nodes = {k: list(nodes[k]) for k in nodes}

    return nodes


def create_network(df, source_field='source', target_field='target', weight='weight'):

    G = nx.Graph()
    nodes = list(set(list(df[source_field].values) + list(df[target_field].values)))
    edges = [(x, y, {'weight': z}) for x, y, z in [tuple(x) for x in df[[source_field, target_field, weight]].values]]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def create_bipartite_network(df, source_field='source', target_field='target', weight='weight'):
    G = nx.Graph()
    G.add_nodes_from(set(df[source_field].values), bipartite=0)
    G.add_nodes_from(set(df[target_field].values), bipartite=1)
    edges = list(zip(df[source_field].values, df[target_field].values, df[weight].apply(lambda x: dict(weight=x))))
    G.add_edges_from(edges)
    return G


def get_bipartite_node_set(network, bipartite=0):
    nodes = set(n for n, d in network.nodes(data=True) if d['bipartite'] == bipartite)
    others = set(network) - nodes
    return list(nodes), list(others)


def create_nx_graph_from_weighted_edges(values, threshold=0.0):  # pylint: disable=unused-argument
    """A simple wrapper for networkx factory function add_weighted_edges_from

    Parameters
    ----------
    values : Edges represented in nx style as a list of (source, target, attributes) triplets

    Returns
    -------
        A new nx.Graph
    """
    G = nx.Graph()
    G.add_weighted_edges_from(values)
    return G


create_network_from_xyw_list = create_nx_graph_from_weighted_edges
