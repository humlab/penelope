from ..interface import LayoutAlgorithm
from ..networkx.networkx_api import nx

engines = ['neato', 'dot', 'circo', 'fdp', 'sfdp']


def layout_network(G, layout_algorithm, **kwargs):

    setup = LAYOUT_SETUP_MAP[layout_algorithm]

    G.graph['K'] = kwargs.get('K', 0.1)
    G.graph['overlap'] = False

    layout = setup.layout_function(G, prog=setup.engine)
    layout = normalize_layout(layout)

    return layout, None


layout_setups = [
    LayoutAlgorithm(
        key='graphviz_{}'.format(engine),
        package='graphviz',
        name='graphviz_{}'.format(engine),
        engine=engine,
        layout_network=layout_network,
        layout_function=nx.nx_pydot.pydot_layout,
        layout_args=lambda **_: {},
    )
    for engine in engines
]

LAYOUT_SETUP_MAP = {x.key: x for x in layout_setups}


def normalize_layout(layout):
    max_xy = max([max(x, y) for x, y in layout.values()])
    layout = {n: (layout[n][0] / max_xy, layout[n][1] / max_xy) for n in layout.keys()}
    return layout
