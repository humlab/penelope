from typing import Any, Callable, Dict, Sequence, Tuple, Union

import bokeh.models as bm
import bokeh.palettes
import networkx as nx
import pandas as pd
import penelope.notebook.widgets_utils as widgets_utils
from bokeh.plotting import figure
from penelope.network import layout_source
from penelope.network.networkx import utility as network_utility

from . import metrics

# pylint: disable=too-many-arguments, unnecessary-lambda


if 'extend' not in globals():
    extend = lambda a, b: a.update(b) or a

layout_algorithms = {
    'Fruchterman-Reingold': lambda x, **args: nx.spring_layout(x, **args),
    'Eigenvectors of Laplacian': lambda x, **args: nx.spectral_layout(x, **args),
    'Circular': lambda x, **args: nx.circular_layout(x, **args),
    'Shell': lambda x, **args: nx.shell_layout(x, **args),
    'Kamada-Kawai': lambda x, **args: nx.kamada_kawai_layout(x, **args),
}


def _layout_args(layout_algorithm: str, network: nx.Graph, scale: float, weight_name: str = 'weight') -> Dict[str, Any]:

    args = None

    if layout_algorithm == 'Shell':
        if nx.is_bipartite(network):
            nodes, other_nodes = network_utility.get_bipartite_node_set(network, bipartite=0)
            args = dict(nlist=[nodes, other_nodes])

    if layout_algorithm == 'Fruchterman-Reingold':
        k = scale  # / math.sqrt(network.number_of_nodes())
        args = dict(dim=2, k=k, iterations=20, weight=weight_name, scale=0.5)

    if layout_algorithm == 'Kamada-Kawai':
        args = dict(dim=2, weight=weight_name, scale=1.0)

    if layout_algorithm == "Circular":
        args = dict(dim=2, center=None, scale=1.0)

    return args


def _get_layout_algorithm(layout_algorithm: str) -> Callable:
    if layout_algorithm not in layout_algorithms:
        raise Exception("Unknown algorithm {}".format(layout_algorithm))
    return layout_algorithms.get(layout_algorithm, None)


def _project_series_to_range(series: pd.Series, low: float, high: float) -> pd.Series:
    norm_series = series / series.max()
    return norm_series.apply(lambda x: low + (high - low) * x)


def project_to_range(value: float, low: float, high: float) -> float:
    """Project a single value to a range (low, high)"""
    return low + (high - low) * value


def project_values_to_range(values: Sequence[float], low: float, high: float) -> Sequence[float]:
    w_max = max(values)
    return [low + (high - low) * (x / w_max) for x in values]


def _plot_network(
    network: nx.Graph,
    layout_algorithm: str = None,
    scale: float = 1.0,
    threshold: float = 0.0,
    node_description: pd.Series = None,
    node_proportions: Union[int, str] = None,
    weight_name: str = 'weight',
    weight_scale: float = 5.0,
    normalize_weights: bool = True,
    node_opts=None,
    line_opts=None,
    element_id: str = 'nx_id3',
    figsize: Tuple[int, int] = (900, 900),
    node_range: Tuple[int, int] = (20, 60),
    edge_range: Tuple[int, int] = (1.0, 5.0),
):
    if threshold > 0:

        values = nx.get_edge_attributes(network, weight_name).values()
        max_weight = max(1.0, max(values))

        print('Max weigth: {}'.format(max_weight))
        print('Mean weigth: {}'.format(sum(values) / len(values)))

        filter_edges = [(u, v) for u, v, d in network.edges(data=True) if d[weight_name] >= (threshold * max_weight)]

        sub_network = network.edge_subgraph(filter_edges)
    else:
        sub_network = network

    args = _layout_args(layout_algorithm, sub_network, scale)
    layout = (_get_layout_algorithm(layout_algorithm))(sub_network, **args)
    lines_source = layout_source.create_edges_layout_data_source(
        sub_network,
        layout,
        weight=weight_name,
        scale=weight_scale,
        normalize=normalize_weights,
        project_range=edge_range,
    )

    nodes_source = layout_source.create_nodes_data_source(sub_network, layout)
    nodes_community = metrics.compute_partition(sub_network)
    community_colors = metrics.partition_colors(nodes_community, bokeh.palettes.Category20[20])

    nodes_source.add(nodes_community, 'community')
    nodes_source.add(community_colors, 'community_color')

    nodes_size = 5
    if node_proportions is not None:
        if isinstance(node_proportions, int):
            nodes_size = node_proportions
        else:
            nodes_size = 'size'
            nodes_weight = project_values_to_range([node_proportions[n] for n in sub_network.nodes], *node_range)
            nodes_source.add(nodes_weight, nodes_size)

    node_opts = extend(dict(color='green', alpha=1.0), node_opts or {})
    line_opts = extend(dict(color='black', alpha=0.4), line_opts or {})

    p = figure(plot_width=figsize[0], plot_height=figsize[1], x_axis_type=None, y_axis_type=None)

    _ = p.multi_line('xs', 'ys', line_width='weights', level='underlay', source=lines_source, **line_opts)
    r_nodes = p.circle('x', 'y', size=nodes_size, source=nodes_source, **node_opts)

    p.add_tools(
        bokeh.models.HoverTool(
            renderers=[r_nodes],
            tooltips=None,
            callback=widgets_utils.glyph_hover_callback2(
                nodes_source, 'node_id', text_ids=node_description.index, text=node_description, element_id=element_id
            ),
        )
    )

    text_opts = dict(x='x', y='y', text='name', level='overlay', x_offset=0, y_offset=0, text_font_size='12pt')

    r_nodes.glyph.fill_color = 'lightgreen'  # 'community_color'

    p.add_layout(
        bm.LabelSet(source=nodes_source, text_align='center', text_baseline='middle', text_color='black', **text_opts)
    )

    return p


def layout_args(layout_algorithm: str, network: nx.Graph, scale: float) -> Dict[str, Any]:
    return _layout_args(layout_algorithm, network, scale)


def get_layout_algorithm(layout_algorithm: str) -> Callable:
    return _get_layout_algorithm(layout_algorithm)


def project_series_to_range(series: pd.Series, low: float, high: float) -> pd.Series:
    return _project_series_to_range(series, low, high)


plot_network = _plot_network
plot_network = _plot_network
