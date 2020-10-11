import bokeh.models as bm
import networkx as nx


def get_edge_layout_data(network, layout, weight='weight'):

    data = [(u, v, d[weight], [layout[u][0], layout[v][0]], [layout[u][1], layout[v][1]])
            for u, v, d in network.edges(data=True)]

    return zip(*data)


# FIXA Merge these two methods, return dict instead (lose bokeh dependency)
def get_edges_source(
    network, layout, scale=1.0, normalize=False, weight='weight', project_range=None, discrete_divisor=None
):

    _, _, weights, xs, ys = get_edge_layout_data(network, layout, weight=weight)
    if isinstance(discrete_divisor, int):
        weights = [max(1, x // discrete_divisor) for x in weights]
    # elif project_range is not None:
    #     # same as _project_series_to_range
    #     w_max = max(weights)
    #     low, high = project_range
    #     weights = [ low + (high - low) * (x / w_max) for x in  weights ]
    elif project_range is not None:
        # same as _project_series_to_range
        w_max = max(weights)
        low, high = project_range
        weights = [int(round(max(low, high * (x / w_max)))) for x in weights]
    else:
        norm = max(weights) if normalize else 1.0
        weights = [scale * x / norm for x in weights]

    lines_source = bm.ColumnDataSource(dict(xs=xs, ys=ys, weights=weights))
    return lines_source


def get_node_subset_source(network, layout, node_list=None):  # pylint: disable=unused-argument

    layout_items = layout.items() if node_list is None else [x for x in layout.items() if x[0] in node_list]

    nodes, nodes_coordinates = zip(*sorted(layout_items))
    xs, ys = list(zip(*nodes_coordinates))

    nodes_source = bm.ColumnDataSource(dict(x=xs, y=ys, name=nodes, node_id=nodes))
    return nodes_source


def create_nodes_data_source(network, layout):  # pylint: disable=unused-argument

    nodes, nodes_coordinates = zip(*sorted([x for x in layout.items()]))  # if x[0] in line_nodes]))
    nodes_xs, nodes_ys = list(zip(*nodes_coordinates))
    nodes_source = bm.ColumnDataSource(dict(x=nodes_xs, y=nodes_ys, name=nodes, node_id=nodes))
    return nodes_source


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


def create_network_from_xyw_list(values, threshold=0.0):  # pylint: disable=unused-argument
    G = nx.Graph()
    G.add_weighted_edges_from(values)
    return G
