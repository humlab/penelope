from typing import List, Sequence

import ipycytoscape
import pandas as pd
from IPython.display import display as IPython_display
from ipywidgets import HTML, Button, Checkbox, Dropdown, HBox, IntSlider, Layout, Output, ToggleButton, VBox

from penelope.plot import get_color_palette

from .display_table import UnnestedExplodeTableDisplayer

# pylint: disable=too-many-instance-attributes

DEFAULT_LAYOUT_ARGUMENTS = {
    'cola': {'maxSimulationTime': 10000},
    'springy': {'stiffness': 400, 'repulsion': 400, 'damping': 0.5},
    'ngraph.forcelayout': {
        'springLength': 100,
        'springCoeff': 0.0008,
        'gravity': -1.2,
        'theta': 0.8,
        'dragCoeff': 0.02,
        'timeStep': 20,
        'iterations': 10000,
        'fit': True,
        'stableThreshold': 0.000009,
    },
}

DEFAULT_CATEGORY_NODE_STYLE = {
    'content': 'data(id)',
    'text-valign': 'center',
}

DEFAULT_TOKEN_NODE_STYLE = {
    'content': 'data(label)',
    'width': 1,
    'height': 1,
    'opacity': 1,  # default 0.8, perf.
    'font-size': 9,
    'font-weight': 'bold',
    'min-zoomed-font-size': 5,  # default 5, higher value gives better performance
    'text-wrap': 'wrap',
    'text-max-width': 50,
    'text-valign': 'center',
    'text-halign': 'center',
    'text-events': 'no',  # default yes
    'color': 'black',
    'text-outline-width': 1,  # default 1
    'text-outline-color': '#fff',
    'text-outline-opacity': 1,
    'overlay-color': '#fff',
}

DEFAULT_EDGE_STYLE = {
    'width': 2,
    'curve-style': 'unbundled-bezier',
    'z-index': 0,
    'overlay-opacity': 0,
}


def css_styles(categories: List[int], custom_styles: dict) -> dict:

    custom_styles = custom_styles or {}
    styles = [
        {
            'selector': 'node[node_type = "category"]',
            'css': {**DEFAULT_CATEGORY_NODE_STYLE, **custom_styles.get('category_nodes', {})},
        },
        {
            'selector': 'node[node_type = "token"]',
            'css': {**DEFAULT_TOKEN_NODE_STYLE, **custom_styles.get('token_nodes', {})},
        },
        {'selector': 'edge', 'style': {**DEFAULT_EDGE_STYLE, **custom_styles.get('edges', {})}},
    ]

    colors = get_color_palette()

    for category_id in categories:
        color = next(colors)
        styles.extend(
            [
                {"selector": f'node[category_id = "{category_id}"]', "css": {"background-color": color}},
                {"selector": f'edge[category_id = "{category_id}"]', "style": {"line-color": color}},
                {'selector': f'node[node_type = "token"][category_id = "{category_id}"]', 'css': {'color': color}},
            ]
        )

    return styles


def create_network(co_occurrences: pd.DataFrame, category_column_name: str) -> ipycytoscape.CytoscapeWidget:

    unique_tokens = co_occurrences['token'].unique().tolist()
    unique_categories = co_occurrences[category_column_name].unique().tolist()

    token_nodes = [
        ipycytoscape.Node(
            data={
                "id": token,
                "label": token,
                "node_type": "token",
                "token_id": token,
            }
        )
        for token in unique_tokens
    ]
    category_nodes = [
        ipycytoscape.Node(
            data={
                "id": str(category),
                "label": str(category),
                "node_type": "category",
                "category_id": str(category),
            }
        )
        for category in unique_categories
    ]
    edges = [
        ipycytoscape.Edge(
            data={
                "id": f"{edge[category_column_name]}_{edge['token']}",
                "source": str(edge[category_column_name]),
                "target": edge['token'],
                "weight": 10.0 * edge['count'],
                "category_id": str(edge[category_column_name]),
            }
        )
        for edge in co_occurrences[[category_column_name, 'token', 'count']].to_dict('records')
    ]
    w = ipycytoscape.CytoscapeWidget(
        layout={'height': '800px'},
        pixelRatio=1.0,
    )
    w.graph.add_nodes(category_nodes)
    w.graph.add_nodes(token_nodes)
    w.graph.add_edges(edges)
    return w


class NetworkDisplayer(UnnestedExplodeTableDisplayer):
    """Probes the token column and explodes it to multiple columns if it contains token-pairs and/or PoS-tags"""

    def __init__(self, name: str = "Network", **opts):
        super().__init__(name=name, **opts)
        self.network: ipycytoscape.CytoscapeWidget = None

        self._view: Output = Output()

        self._node_spacing: IntSlider = IntSlider(description='', min=3, max=500, value=50, layout={'width': '200px'})
        self._edge_length_val: IntSlider = IntSlider(
            description='', min=3, max=500, value=50, layout={'width': '200px'}
        )
        self._padding: IntSlider = IntSlider(description='', min=3, max=500, value=50, layout={'width': '200px'})
        self._label: HTML = HTML(value='&nbsp;', layout={'width': '200px'})
        self._network_layout = Dropdown(
            description='',
            options=['cola', 'klay', 'circle', 'concentric'],
            value='cola',
            layout={'width': '115px'},
        )
        self._relayout = Button(
            description="Continue", button_style='Info', layout=Layout(width='115px', background_color='blue')
        )
        self._animate: Checkbox = ToggleButton(
            description="Animate",
            icon='check',
            value=True,
            layout={'width': '115px'},
        )
        self._curve_style = Dropdown(
            description='',
            options=[
                ('Straight line', 'haystack'),
                ('Curve, Bezier', 'bezier'),
                ('Curve, Bezier*', 'unbundled-bezier'),
            ],
            value='haystack',
            layout={'width': '115px'},
        )

        self._custom_styles: dict = None

        self._buzy: bool = False

    def setup(self, *_, **__):
        # self._custom_styles = custom_styles()
        self._network_layout.observe(self._layout_handler, names='value')
        self._animate.observe(self._toggle_state_changed, 'value')
        self._relayout.on_click(self._relayout_handler)

    def plot(self, *, data: Sequence[pd.DataFrame], temporal_key: str, **_) -> None:
        """Data is unstacked i.e. columns are token@pivot_keys"""
        unstacked_data: pd.DataFrame = data if isinstance(data, pd.DataFrame) else data[-1]

        if temporal_key not in unstacked_data.columns:
            unstacked_data[temporal_key] = unstacked_data.index

        network_data: pd.DataFrame = self.create_data_frame(unstacked_data, temporal_key)

        if network_data is None:
            self.alert("No data!")
            return self

        self.network = create_network(network_data, temporal_key)
        self.set_layout()
        self.network.set_style(css_styles(network_data[temporal_key].unique(), self.custom_styles))

        with self.output:
            IPython_display(self.layout())

        with self._view:
            IPython_display(self.network)

        return self

    def _relayout_handler(self, *_):
        if self.network:
            self.network.relayout()

    def set_layout(self):

        self.alert("Layout: " + self.network_layout)

        if not self.network:
            return

        self.network.set_layout(
            name=self.network_layout,
            animate=self.animate,
            **DEFAULT_LAYOUT_ARGUMENTS.get(self.network_layout, {}),
        )

    def _layout_handler(self, *_):
        self.set_layout()

    def lock(self, value: bool = True) -> None:
        self._buzy = not value

        self._relayout.disabled = value
        self._curve_style.disabled = value
        self._network_layout.disabled = value
        self._animate.disabled = value

    def alert(self, msg: str = '&nbsp;') -> None:
        self._label.value = msg or '&nbsp;'

    def _toggle_state_changed(self, event):
        event['owner'].icon = 'check' if event['new'] else ''

    def layout(self):
        return VBox(
            [
                HBox(
                    [
                        VBox(
                            [
                                HTML("<b>Layout</b>"),
                                self._network_layout,
                            ]
                        ),
                        VBox(
                            [
                                HTML("<b>Curve style</b>"),
                                self._curve_style,
                            ]
                        ),
                        VBox(
                            [
                                self._animate,
                                self._relayout,
                            ]
                        ),
                        VBox(
                            [
                                HTML("&nbsp;"),
                                self._label,
                            ]
                        ),
                    ]
                ),
                self._view,
            ]
        )

    @property
    def network_layout(self) -> List[int]:
        return self._network_layout.value

    @property
    def curve_style(self) -> str:
        return self._curve_style.value

    @property
    def animate(self) -> bool:
        return self._animate.value

    @property
    def custom_styles(self):
        style = {}
        if style.get('edges', None) is None:
            style['edges'] = {}
        style['edges']['curve-style'] = self.curve_style
        return style
