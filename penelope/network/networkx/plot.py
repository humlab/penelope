import networkx as nx
import pydotplus
from IPython.display import Image


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


def plot(G, **kwargs):  # pylint: disable=unused-argument

    P = nx.nx_pydot.to_pydot(G)
    P.format = 'svg'
    # if root is not None :
    #    P.set("root",make_str(root))
    D = P.create_dot(prog='circo')
    if D == "":
        return None
    Q = pydotplus.graph_from_dot_data(D)

    image = Image(Q.create_png())
    return image
