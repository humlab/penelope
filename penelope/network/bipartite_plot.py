from __future__ import annotations

from typing import Any, Sequence

import bokeh.models as bm
import bokeh.plotting as bp
import pandas as pd

from penelope.notebook import widgets_utils as wu

from . import layout_source, metrics, plot_utility
from .networkx import utility as nu
from .networkx.networkx_api import nx

# pylint: disable=unused-argument, too-many-locals


def plot_bipartite_dataframe(
    data: pd.DataFrame,
    network_layout: str,
    *,
    scale: float,
    titles: pd.DataFrame,
    source_name: str,
    target_name: str,
    element_id: str,
) -> None:
    network: nx.Graph = nu.create_bipartite_network(
        data[[target_name, source_name, 'weight']], target_name, source_name
    )
    args: dict[str, Any] = plot_utility.layout_args(network_layout, network, scale)
    layout: nu.NodesLayout = (plot_utility.layout_algorithms[network_layout])(network, **args)
    p = plot_bipartite_network(network, layout, scale=scale, titles=titles, element_id=element_id)
    bp.show(p)


def plot_highlighted_bipartite_dataframe(
    network_data: pd.DataFrame,
    network_layout: str,
    highlight_topic_ids: Sequence[int],
    titles: pd.DataFrame,
    scale: float,
    source_name: str,
    target_name: str,
    element_id: str,
):
    network: nx.Graph = nu.create_bipartite_network(network_data, source_name, target_name)
    args = plot_utility.layout_args(network_layout, network, scale)
    layout_data = (plot_utility.layout_algorithms[network_layout])(network, **args)
    p = plot_bipartite_network(
        network,
        layout_data,
        scale=scale,
        titles=titles,
        highlight_topic_ids=highlight_topic_ids,
        element_id=element_id,
    )

    bp.show(p)


def plot_bipartite_network(
    network: nx.Graph,
    layout_data: nu.NodesLayout,
    scale: float = 1.0,
    titles: pd.DataFrame = None,
    highlight_topic_ids=None,
    element_id: str = 'nx_id1',
    width: int = 1000,
    height: int = 600,
) -> bp.figure:
    """Plot a bipartite network. Return bokeh.figure"""
    tools: str = 'pan,wheel_zoom,box_zoom,reset,hover,save'

    source_nodes, target_nodes = nu.get_bipartite_node_set(network, bipartite=0)

    color_map: dict = (
        {x: 'brown' if x in highlight_topic_ids else 'skyblue' for x in target_nodes}
        if highlight_topic_ids is not None
        else None
    )
    color_specifier: str = 'colors' if highlight_topic_ids is not None else 'skyblue'

    source_source: bm.ColumnDataSource = layout_source.create_nodes_subset_data_source(
        network, layout_data, source_nodes
    )
    target_source: bm.ColumnDataSource = layout_source.create_nodes_subset_data_source(
        network, layout_data, target_nodes, color_map=color_map
    )
    lines_source: bm.ColumnDataSource = layout_source.create_edges_layout_data_source(
        network, layout_data, scale=6.0, normalize=False
    )

    edges_alphas: list[float] = metrics.compute_alpha_vector(lines_source.data['weights'])

    lines_source.add(edges_alphas, 'alphas')

    p: bp.figure = bp.figure(
        width=width,
        height=height,
        sizing_mode='scale_width',
        x_axis_type=None,
        y_axis_type=None,
        tools=tools,
    )

    _ = p.multi_line(
        xs='xs', ys='ys', line_width='weights', level='underlay', alpha='alphas', color='black', source=lines_source
    )
    _ = p.circle(x='x', y='y', size=40, source=source_source, color='lightgreen', line_width=1, alpha=1.0)

    r_targets: bm.GlyphRenderer = p.circle(
        x='x', y='y', size=25, source=target_source, color=color_specifier, alpha=1.0
    )

    p.add_tools(
        bm.HoverTool(
            renderers=[r_targets],
            tooltips=None,
            callback=wu.glyph_hover_callback2(
                glyph_source=target_source,
                glyph_id='node_id',
                text_ids=titles.index,
                text=titles,
                element_id=element_id,
            ),
        )
    )

    text_opts: dict = dict(x='x', y='y', text='name', level='overlay', x_offset=0, y_offset=0, text_font_size='8pt')

    p.add_layout(
        bm.LabelSet(source=source_source, text_color='black', text_align='center', text_baseline='middle', **text_opts)
    )

    target_source.data['name'] = [str(x) for x in target_source.data['name']]  # pylint: disable=unsubscriptable-object

    p.add_layout(
        bm.LabelSet(source=target_source, text_color='black', text_align='center', text_baseline='middle', **text_opts)
    )

    return p
