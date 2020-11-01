import bokeh.models
import bokeh.palettes
import penelope.network.layout as network_layout
import penelope.network.networkx.utility as networkx_utility
import penelope.notebook.widgets_utils as widgets_utils
import penelope.utility as utility
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

# pylint: disable=too-many-arguments

TOOLS = "pan,wheel_zoom,box_zoom,reset,previewsave"

DFLT_PALETTE = bokeh.palettes.Set3[12]
DFLT_FIG_OPTS = dict(plot_height=900, plot_width=900, tools=TOOLS)

DFLT_NODE_OPTS = dict(color='green', level='overlay', alpha=1.0)

DFLT_EDGE_OPTS = dict(color='black', alpha=0.2)

DFLT_TEXT_OPTS = dict(x='x', y='y', text='name', level='overlay', text_align='center', text_baseline='middle')

DFLT_LABEL_OPTS = dict(
    level='overlay',
    text_align='center',
    text_baseline='middle',
    render_mode='canvas',
    text_font="Tahoma",
    text_font_size="9pt",
    text_color='black',
)


def get_palette(name):

    if name not in bokeh.palettes.all_palettes.keys():
        return bokeh.palettes.RdYlBu[11]

    key = max(bokeh.palettes.all_palettes[name].keys())

    return bokeh.palettes.all_palettes[name][key]


def setup_node_size(nodes, node_size, node_size_range):

    if node_size is None:
        node_size = node_size_range[0]

    if node_size in nodes.keys() and node_size_range is not None:
        nodes['clamped_size'] = utility.clamp_values(nodes[node_size], node_size_range)
        node_size = 'clamped_size'

    return node_size


def adjust_node_label_offset(nodes, node_size, default_y_offset=5):

    label_x_offset = 0
    label_y_offset = 'y_offset' if node_size in nodes.keys() else node_size + default_y_offset
    if label_y_offset == 'y_offset':
        nodes['y_offset'] = [
            y + r for (y, r) in zip(nodes['y'], [r / 2.0 + default_y_offset for r in nodes[node_size]])
        ]
    return label_x_offset, label_y_offset


def plot(  # pylint: disable=W0102
    network,
    layout,
    scale=1.0,  # pylint: disable=unused-argument
    threshold=0.0,
    node_description=None,
    node_size=5,
    node_size_range=[20, 40],
    weight_scale=5.0,
    normalize_weights=True,
    node_opts=None,
    line_opts=None,
    text_opts=None,
    element_id='nx_id3',
    figsize=(900, 900),
    tools=None,
    palette=DFLT_PALETTE,  # pylint: disable=unused-argument
    **figkwargs,
):
    if threshold > 0:
        network = networkx_utility.get_sub_network(network, threshold)

    edges = networkx_utility.get_positioned_edges(network, layout)

    if normalize_weights and 'weight' in edges.keys():
        max_weight = max(edges['weight'])
        edges['weight'] = [float(x) / max_weight for x in edges['weight']]

    if weight_scale != 1.0 and 'weight' in edges.keys():
        edges['weight'] = [weight_scale * float(x) for x in edges['weight']]

    # edges = dict(source=u, target=v, xs=xs, ys=ys, weights=weights)

    nodes = networkx_utility.get_positioned_nodes(network, layout)

    # node_size = setup_node_size(nodes, node_size, node_size_range)
    if node_size in nodes.keys() and node_size_range is not None:
        nodes['clamped_size'] = utility.clamp_values(nodes[node_size], node_size_range)
        node_size = 'clamped_size'

    label_y_offset = 'y_offset' if node_size in nodes.keys() else node_size + 8
    if label_y_offset == 'y_offset':
        nodes['y_offset'] = [y + r for (y, r) in zip(nodes['y'], [r / 2.0 + 8 for r in nodes[node_size]])]

    edges = {k: list(edges[k]) for k in edges}
    nodes = {k: list(nodes[k]) for k in nodes}

    edges_source = bokeh.models.ColumnDataSource(edges)
    nodes_source = bokeh.models.ColumnDataSource(nodes)

    node_opts = utility.extend(DFLT_NODE_OPTS, node_opts or {})
    line_opts = utility.extend(DFLT_EDGE_OPTS, line_opts or {})

    p = figure(plot_width=figsize[0], plot_height=figsize[1], tools=tools or TOOLS, **figkwargs)

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    _ = p.multi_line(
        xs='xs', ys='ys', line_width='weights', source=edges_source, **line_opts
    )  # pylint: disable=too-many-function-args
    r_nodes = p.circle(
        x='x', y='y', size=node_size, source=nodes_source, **node_opts
    )  # pylint: disable=too-many-function-args

    if 'fill_color' in nodes.keys():
        r_nodes.glyph.fill_color = 'fill_color'

    if node_description is not None:
        text_source = ColumnDataSource(dict(text_id=node_description.index, text=node_description))
        p.add_tools(
            bokeh.models.HoverTool(
                renderers=[r_nodes],
                tooltips=None,
                callback=widgets_utils.glyph_hover_callback(
                    nodes_source, 'node_id', text_source, element_id=element_id
                ),
            )
        )

    label_opts = {
        **DFLT_TEXT_OPTS,
        **dict(y_offset=label_y_offset, text_color='black', text_baseline='bottom'),
        **(text_opts or {}),
    }

    p.add_layout(bokeh.models.LabelSet(source=nodes_source, **label_opts))

    return p


def plot_network(nodes, edges, plot_opts, fig_opts=None):

    edges_source = bokeh.models.ColumnDataSource(edges)
    nodes_source = bokeh.models.ColumnDataSource(nodes)

    node_opts = utility.extend(DFLT_NODE_OPTS, plot_opts.get('node_opts', {}))
    line_opts = utility.extend(DFLT_EDGE_OPTS, plot_opts.get('line_opts', {}))
    fig_opts = utility.extend(DFLT_FIG_OPTS, fig_opts or {})

    node_size = plot_opts.get('node_size', 1)

    p = figure(**fig_opts)

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    if 'line_color' in edges.keys():
        line_opts = utility.extend(line_opts, {'line_color': 'line_color', 'alpha': 1.0})

    _ = p.multi_line(
        xs='xs', ys='ys', line_width='weight', source=edges_source, **line_opts
    )  # pylint: disable=too-many-function-args
    r_nodes = p.circle(
        x='x', y='y', size=node_size, source=nodes_source, **node_opts
    )  # pylint: disable=too-many-function-args

    if 'fill_color' in nodes.keys():
        r_nodes.glyph.fill_color = 'fill_color'

    node_description = plot_opts.get('node_description', None)
    if node_description is not None:
        element_id = plot_opts.get('element_id', '_')
        text_source = ColumnDataSource(dict(text_id=node_description.index, text=node_description))
        p.add_tools(
            bokeh.models.HoverTool(
                renderers=[r_nodes],
                tooltips=None,
                callback=widgets_utils.glyph_hover_callback(
                    nodes_source, 'node_id', text_source=text_source, element_id=element_id
                ),
            )
        )

    node_label = plot_opts.get('node_label', None)
    if node_label is not None and node_label in nodes.keys():
        label_opts = {**DFLT_LABEL_OPTS, **plot_opts.get('node_label_opts', {})}
        p.add_layout(bokeh.models.LabelSet(source=nodes_source, x='x', y='y', text=node_label, **label_opts))

    edge_label = plot_opts.get('edge_label', None)
    if edge_label is not None and edge_label in edges.keys():
        label_opts = {**DFLT_LABEL_OPTS, **plot_opts.get('edge_label_opts', {})}
        p.add_layout(bokeh.models.LabelSet(source=edges_source, x='m_x', y='m_y', text=edge_label, **label_opts))

    handle = bokeh.plotting.show(p, notebook_handle=True)

    return dict(handle=handle, edges=edges, nodes=nodes, edges_source=edges_source, nodes_source=nodes_source)


def plot_df(df, source='source', target='target', weight='weight', layout_opts=None, plot_opts=None, fig_opts=None):

    # print([ x.key for x in network_layout.layout_setups])

    g = networkx_utility.df_to_nx(df, source=source, target=target, bipartite=False, weight=weight)

    layout, _ = network_layout.layout_network(g, layout_opts['algorithm'], **layout_opts['args'])

    edges = networkx_utility.get_positioned_edges2(g, layout)
    nodes = networkx_utility.get_positioned_nodes(g, layout)

    plot_data = plot_network(nodes=nodes, edges=edges, plot_opts=plot_opts, fig_opts=fig_opts)

    return plot_data
