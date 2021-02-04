from typing import Tuple

import bokeh.models as bm
import networkx as nx

from .networkx.utility import layout_edges


def create_edges_layout_data_source(
    network: nx.Graph,
    layout,
    scale: float = 1.0,
    normalize: bool = False,
    weight: str = 'weight',
    project_range: Tuple[float, float] = None,
    discrete_divisor=None,
) -> bm.ColumnDataSource:

    _, _, weights, xs, ys = layout_edges(network, layout, weight=weight)
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


def create_nodes_subset_data_source(
    network: nx.Graph, layout, node_list=None, color_map=None  # pylint: disable=unused-argument
) -> bm.ColumnDataSource:

    layout_items = layout.items() if node_list is None else [x for x in layout.items() if x[0] in node_list]

    nodes, nodes_coordinates = zip(*sorted(layout_items))
    xs, ys = list(zip(*nodes_coordinates))

    nodes_source = bm.ColumnDataSource(dict(x=xs, y=ys, name=nodes, node_id=nodes))
    if color_map is not None:
        nodes_source.add([color_map[x] for x in nodes], "colors")
    return nodes_source


def create_nodes_data_source(network: nx.Graph, layout) -> bm.ColumnDataSource:  # pylint: disable=unused-argument

    nodes, nodes_coordinates = zip(*sorted([x for x in layout.items()]))
    nodes_xs, nodes_ys = list(zip(*nodes_coordinates))
    nodes_source = bm.ColumnDataSource(dict(x=nodes_xs, y=nodes_ys, name=nodes, node_id=nodes))
    return nodes_source
