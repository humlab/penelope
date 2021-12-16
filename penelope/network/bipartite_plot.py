from typing import List

import bokeh.models as bm
import bokeh.plotting as bp
import networkx as nx
import pandas as pd
from penelope.network import layout_source, metrics
from penelope.network.networkx import utility as network_utility
from penelope.notebook import widgets_utils

# pylint: disable=unused-argument, too-many-locals


def plot_bipartite_network(
    network: nx.Graph,
    layout_data: network_utility.NodesLayout,
    scale: float = 1.0,
    titles: pd.DataFrame = None,
    highlight_topic_ids=None,
    element_id: str = 'nx_id1',
    plot_width: int = 1000,
    plot_height: int = 600,
) -> bp.Figure:
    """Plot a bipartite network. Return bokeh.Figure"""
    tools: str = 'pan,wheel_zoom,box_zoom,reset,hover,save'

    source_nodes, target_nodes = network_utility.get_bipartite_node_set(network, bipartite=0)

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

    edges_alphas: List[float] = metrics.compute_alpha_vector(lines_source.data['weights'])

    lines_source.add(edges_alphas, 'alphas')

    p: bp.Figure = bp.figure(
        plot_width=plot_width, plot_height=plot_height, x_axis_type=None, y_axis_type=None, tools=tools
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
            callback=widgets_utils.glyph_hover_callback2(
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
