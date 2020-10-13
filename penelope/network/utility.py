import bokeh.models as bm

from .networkx import utility as nx_utils


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


# FIXME; #4 Consolidate network utility functions (utiity vs networkx.utility)
create_bipartite_network = nx_utils.create_bipartite_network
get_bipartite_node_set = nx_utils.get_bipartite_node_set
create_network = nx_utils.create_network
create_network_from_xyw_list = nx_utils.create_nx_graph_from_weighted_edges
get_edge_layout_data = nx_utils.get_edge_layout_data
