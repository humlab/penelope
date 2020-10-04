import math

from .graphtool.layout import layout_setups as gt_layout_setups
from .graphviz.layout import layout_setups as gv_layout_setups
from .networkx.layout import layout_setups as nx_layout_setups

layout_setups = nx_layout_setups + gv_layout_setups + gt_layout_setups

layouts = {x.key: x for x in layout_setups}


def layout_network(G, layout_algorithm, **kwargs):

    return layouts[layout_algorithm].layout_network(G, layout_algorithm=layout_algorithm, **kwargs)


def adjust_edge_endpoint(p, q, d):

    dx, dy = q[0] - p[0], q[1] - p[1]
    alpha = math.atan2(dy, dx)
    w = (q[0] - d * math.cos(alpha), q[1] - d * math.sin(alpha))
    return w


def adjust_edge_lengths(edges, nodes):

    node2id = {x: i for i, x in enumerate(nodes['name'])}

    for i in range(0, len(edges['xs'])):

        p1 = (edges['xs'][i][0], edges['ys'][i][0])
        p2 = (edges['xs'][i][1], edges['ys'][i][1])

        source_id = node2id[edges['source'][i]]
        target_id = node2id[edges['target'][i]]

        x1, y1 = adjust_edge_endpoint(p1, p2, nodes['size'][source_id])
        x2, y2 = adjust_edge_endpoint(p2, p1, nodes['size'][target_id])

        edges['xs'][i][0] = x1
        edges['xs'][i][1] = x2
        edges['ys'][i][0] = y1
        edges['ys'][i][1] = y2
