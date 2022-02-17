import types

from IPython.display import Image
from loguru import logger

from penelope.utility import deprecated

from .networkx_api import nx

try:
    import pydotplus  # pylint: disable=import-error
except ImportError:
    logger.info("pydotplus not installed (skipping plot)")
    pydotplus = False


def apply_styles(graph, styles):
    graph.graph_attr.update(('graph' in styles and styles['graph']) or {})
    graph.node_attr.update(('nodes' in styles and styles['nodes']) or {})
    graph.edge_attr.update(('edges' in styles and styles['edges']) or {})
    return graph


STYLES = {
    'graph': {
        'label': 'Graph',
        'fontsize': '16',
        'fontcolor': 'white',
        'bgcolor': '#333333',
        'rankdir': 'BT',
    },
    'nodes': {
        'fontname': 'Helvetica',
        'shape': 'hexagon',
        'fontcolor': 'white',
        'color': 'white',
        'style': 'filled',
        'fillcolor': '#006699',
    },
    'edges': {
        'style': 'dashed',
        'color': 'white',
        'arrowhead': 'open',
        'fontname': 'Courier',
        'fontsize': '12',
        'fontcolor': 'white',
    },
}


@deprecated
def plot(G: nx.Graph, **kwargs) -> Image:  # pylint: disable=unused-argument

    if not isinstance(pydotplus, types.ModuleType):
        return None

    P = nx.nx_pydot.to_pydot(G)
    P.format = 'svg'
    D = P.create_dot(prog='circo')
    if D == "":
        return None
    Q = pydotplus.graph_from_dot_data(D)
    image = Image(Q.create_png())
    return image
