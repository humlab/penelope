try:
    import graph_tool.all as gt  # type: ignore # pylint: disable=import-error
    import graph_tool.draw as gt_draw  # type: ignore # pylint: disable=import-error, unused-argument
except ImportError:
    gt_draw = None
    gt = None
    print('warning: graph_tool not installed!')


def plot(G_gt, layout_gt, n_range, palette, **kwargs):  # pylint: disable=unused-argument

    v_text = G_gt.vertex_properties['id']
    # v_degrees_p = G_gt.degree_property_map('out')
    # v_degrees_p.a = np.sqrt(v_degrees_p.a)+2
    v_degrees_p = G_gt.vertex_properties['degree']
    v_size_p = gt.prop_to_size(v_degrees_p, n_range[0], n_range[1])
    v_fill_color = G_gt.vertex_properties['fill_color']
    e_weights = G_gt.edge_properties['weight']
    e_size_p = gt.prop_to_size(e_weights, 1.0, 4.0)
    # state = gt.minimize_blockmodel_dl(G_gt)
    # state.draw(
    # c = gt.all.closeness(G_gt)

    v_blocks = gt.minimize_blockmodel_dl(G_gt).get_blocks()
    plot_color = G_gt.new_vertex_property('vector<double>')
    G_gt.vertex_properties['plot_color'] = plot_color

    for v_i, v in enumerate(G_gt.vertices()):
        scolor = palette[v_blocks[v_i]]
        plot_color[v] = tuple(int(scolor[i : i + 2], 16) for i in (1, 3, 5)) + (1,)

    gt_draw.graph_draw(
        G_gt,
        # vorder=c,
        pos=layout_gt,
        output_size=(1000, 1000),
        # vertex_text_offset=[-1,1],
        vertex_text_position=0.0,
        vertex_text=v_text,
        vertex_color=[1, 1, 1, 0],
        vertex_fill_color=v_fill_color,
        vertex_size=v_size_p,
        vertex_font_family='helvetica',
        vertex_text_color='black',
        edge_pen_width=e_size_p,
        inline=True,
    )
