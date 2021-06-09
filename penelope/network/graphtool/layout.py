from penelope.network.graphtool.utility import nx2gt

from ..interface import LayoutAlgorithm

try:
    import graph_tool.all as gt  # type: ignore # pylint: disable=import-error
    import graph_tool.draw as gt_draw  # type: ignore # pylint: disable=import-error, unused-argument
except ImportError:
    gt_draw = None
    gt = None
    print('warning: graph_tool not installed!')


def sfdp_args(G_gt, **kwargs):
    weight = G_gt.edge_properties['weight']
    K = kwargs.get('K', 0.1)
    C = kwargs.get('C', 1.0)
    p = kwargs.get('p', 0.1)
    return dict(eweight=weight, K=K, C=(C / 100.0), gamma=p)


def arf_args(G_gt, **kwargs):
    weight = G_gt.edge_properties['weight']
    K = kwargs.get('K', 0.1)
    C = kwargs.get('C', 1.0)
    return dict(weight=weight, d=K, a=C)


def fruchterman_reingold_args(G_gt, **kwargs):
    weight = G_gt.edge_properties['weight']
    K = kwargs.get('K', 0.1)
    C = kwargs.get('C', 1.0)
    N = len(G_gt)  # !! G
    return dict(weight=weight, a=(2.0 * N * K), r=2.0 * C)


def layout_network(G, **kwargs):

    setup = _get_layout_setup(kwargs['layout_algorithm'])

    g_gt = nx2gt(G)
    g_gt.set_directed(False)

    args = setup.layout_args(g_gt, **kwargs)

    layout_gt = setup.layout_function(g_gt, **args)

    layout = {g_gt.vertex_properties['id'][i]: layout_gt[i] for i in g_gt.vertices()}

    return layout, (g_gt, layout_gt)


layout_setups = (
    []
    if gt is None
    else [
        LayoutAlgorithm(
            key='graphtool_arf',
            package='graphtool',
            name='graph-tool (arf)',
            layout_network=layout_network,
            layout_function=gt_draw.arf_layout,
            layout_args=arf_args,
        ),
        LayoutAlgorithm(
            key='graphtool_sfdp',
            package='graphtool',
            name='graph-tool (sfdp)',
            layout_network=layout_network,
            layout_function=gt_draw.sfdp_layout,
            layout_args=sfdp_args,
        ),
        LayoutAlgorithm(
            key='graphtool_fr',
            package='graphtool',
            name='graph-tool (FR)',
            layout_network=layout_network,
            layout_function=gt_draw.fruchterman_reingold_layout,
            layout_args=fruchterman_reingold_args,
        ),
    ]
)

layout_setup_map = {x.key: x for x in layout_setups}


def _get_layout_setup(key):
    return layout_setup_map[key]
