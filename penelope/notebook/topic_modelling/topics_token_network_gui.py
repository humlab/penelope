import glob
import os
from dataclasses import dataclass, field
from typing import Any, Callable, List

import ipycytoscape
import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from penelope import topic_modelling
from penelope.notebook.ipyaggrid_utility import display_grid
from penelope.plot import get_color_palette
from penelope.utility.filename_fields import FilenameFieldSpecs

view = widgets.Output()


DEFAULT_LAYOUT_ARGUMENTS = {
    'cola': {'maxSimulationTime': 60000},
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
MAX_TOPIC_TOKEN_COUNT = 500


def css_styles(topic_ids: List[int], curve_style: str = 'unbundled-bezier') -> dict:
    styles = [
        {
            'selector': 'node[node_type = "topic"]',
            'css': {
                'content': 'data(id)',
                'text-valign': 'center',
            },
        },
        {
            'selector': 'node[node_type = "token"]',
            'css': {
                'content': 'data(label)',
                'width': 1,
                'height': 1,
                'opacity': 0.8,
                'font-size': 9,
                'font-weight': 'bold',
                'min-zoomed-font-size': 4,
                'text-wrap': 'wrap',
                'text-max-width': 50,
                'text-valign': 'center',
                'text-halign': 'center',
                'text-events': 'yes',
                'color': 'black',
                'text-outline-width': 1,
                'text-outline-color': '#fff',
                'text-outline-opacity': 1,
                'overlay-color': '#fff',
            },
        },
        {
            'selector': 'edge',
            'style': {
                'width': 2,
                'curve-style': curve_style,
                'z-index': 0,
                'overlay-opacity': 0,
            },
        },
    ]

    colors = get_color_palette()

    for topic_id in topic_ids:
        color = next(colors)
        styles.extend(
            [
                {"selector": f'node[topic_id = "{topic_id}"]', "css": {"background-color": color}},
                {"selector": f'edge[topic_id = "{topic_id}"]', "style": {"line-color": color}},
                {'selector': f'node[node_type = "token"][topic_id = "{topic_id}"]', 'css': {'color': color}},
            ]
        )

    return styles


def to_dict(topics_tokens: pd.DataFrame) -> dict:

    unique_topics = topics_tokens.groupby(['topic', 'topic_id']).size().reset_index()[['topic', 'topic_id']]
    unique_tokens = topics_tokens.groupby('token')['topic_id'].apply(list)

    source_network_data = {
        'nodes': [
            {
                "data": {
                    "id": node['topic'],
                    "label": node['topic'],
                    "node_type": "topic",
                    "topic_id": str(node['topic_id']),
                },
            }
            for node in unique_topics.to_dict('records')
        ]
        + [
            {
                "data": {
                    "id": w,
                    "label": w,
                    "node_type": "token",
                    "topic_id": str(unique_tokens[w][0]) if len(unique_tokens[w]) == 1 else "",
                },
            }
            for w in unique_tokens.index
        ],
        'edges': [
            {
                "data": {
                    "id": f"{edge['topic_id']}_{edge['token']}",
                    "source": edge['topic'],
                    "target": edge['token'],
                    "weight": 10.0 * edge['weight'],
                    "topic_id": str(edge['topic_id']),
                },
            }
            for edge in topics_tokens[['topic', 'token', 'topic_id', 'weight']].to_dict('records')
        ],
    }
    return source_network_data


def create_network(topics_tokens: pd.DataFrame) -> ipycytoscape.CytoscapeWidget:
    source_network_data = to_dict(topics_tokens=topics_tokens)
    w = ipycytoscape.CytoscapeWidget(layout={'height': '800px'})
    w.min_zoom = 0.2
    w.max_zoom = 1.5
    w.graph.add_graph_from_json(source_network_data)
    return w


@dataclass
class ViewModel:

    # FIXME: This must be bundled and stored in the InferredTopicsData data!!!
    filename_fields: FilenameFieldSpecs = None

    _topics_data: topic_modelling.InferredTopicsData = field(init=False, default=None)
    _top_topic_tokens: pd.DataFrame = field(init=False, default=None)

    @property
    def top_topic_tokens(self) -> pd.DataFrame:
        return self._top_topic_tokens

    def update(self, data: topic_modelling.InferredTopicsData = None) -> "ViewModel":

        if data is not None:
            self._topics_data = data

        if self._topics_data is None:
            return self

        self._top_topic_tokens = self._topics_data.top_topic_token_weights(MAX_TOPIC_TOKEN_COUNT)

        return self

    def get_topics_tokens(self, topic_ids: List[int], top_count: int) -> pd.DataFrame:
        topics_tokens = self.top_topic_tokens[
            (self.top_topic_tokens.index.isin(topic_ids) & (self.top_topic_tokens.position <= top_count))
        ].reset_index()
        topics_tokens['topic'] = topics_tokens.topic_id.apply(lambda x: f"Topic #{x}")
        return topics_tokens[['topic', 'token', 'weight', 'topic_id', 'position']]

    @property
    def num_topics(self):
        if not self._topics_data:
            return 0
        return self._topics_data.num_topics


def find_inferred_models(folder: str) -> List[str]:
    """Return YAML filenames in `folder`"""
    filenames = glob.glob(os.path.join(folder, "**/*document_topic_weights.zip"), recursive=True)
    folders = [os.path.split(filename)[0] for filename in filenames]
    return folders


@view.capture(clear_output=False)
def default_loader(folder: str, filename_fields: Any = None) -> topic_modelling.InferredTopicsData:
    data = topic_modelling.InferredTopicsData.load(folder=folder, filename_fields=filename_fields)
    return data


@view.capture(clear_output=True)
def default_displayer(opts: "GUI") -> None:

    if opts.model.top_topic_tokens is None:
        return

    topics_tokens = opts.model.get_topics_tokens(opts.topics_ids, opts.top_count)

    if opts.output_format == "network":
        network = create_network(topics_tokens)
        opts.network = network
        opts.set_layout()
        style = css_styles(topics_tokens.topic_id.unique(), opts.curve_style)
        network.set_style(style)
        display(network)
        return

    if opts.output_format == "table":
        g = display_grid(topics_tokens)
        display(g)

    if opts.output_format == "gephi":
        topics_tokens = topics_tokens[['topic', 'token', 'weight']]
        topics_tokens.columns = ['Source', 'Target', 'Weight']
        g = display_grid(topics_tokens)
        display(g)


# pylint: disable=too-many-instance-attributes
@dataclass
class GUI:

    network: ipycytoscape.CytoscapeWidget = None

    _source_folder: widgets.Dropdown = widgets.Dropdown(layout={'width': '200px'})
    _topic_ids: widgets.SelectMultiple = widgets.SelectMultiple(
        description="", options=[], value=[], rows=8, layout={'width': '100px'}
    )
    _top_count: widgets.IntSlider = widgets.IntSlider(
        description='', min=3, max=200, value=50, layout={'width': '200px'}
    )

    _node_spacing: widgets.IntSlider = widgets.IntSlider(
        description='', min=3, max=500, value=50, layout={'width': '200px'}
    )
    _edge_length_val: widgets.IntSlider = widgets.IntSlider(
        description='', min=3, max=500, value=50, layout={'width': '200px'}
    )
    _padding: widgets.IntSlider = widgets.IntSlider(description='', min=3, max=500, value=50, layout={'width': '200px'})
    _label: widgets.HTML = widgets.HTML(value='&nbsp;', layout={'width': '200px'})
    _output_format = widgets.Dropdown(
        description='', options=['network', 'table', 'gephi'], value='network', layout={'width': '200px'}
    )
    _network_layout = widgets.Dropdown(
        description='',
        options=[
            'cola',
            'klay',
            'circle',
            'concentric',
            # 'cise',
            # 'springy',
            # 'ngraph.forcelayout',
            # 'cose-bilkent', 'cose', 'euler', 'fcose', 'spread', 'elk', 'stress', 'force', 'avsdf',
        ],
        value='cola',
        layout={'width': '115px'},
    )
    _button = widgets.Button(
        description="Display", button_style='Success', layout=widgets.Layout(width='115px', background_color='blue')
    )
    _relayout = widgets.Button(
        description="Continue", button_style='Info', layout=widgets.Layout(width='115px', background_color='blue')
    )
    _animate: widgets.Checkbox = widgets.Checkbox(description='Animate', value=False)
    _curve_style = widgets.Dropdown(
        description='',
        options=[
            'haystack',
            'unbundled-bezier',
        ],
        value='haystack',
        layout={'width': '115px'},
    )
    model: ViewModel = None

    loader: Callable[[str], topic_modelling.InferredTopicsData] = None
    displayer: Callable[["GUI"], None] = None

    @view.capture(clear_output=False)
    def _displayer(self, *_):
        if not self.displayer:
            return
        self.alert('<b>Computing</b>...')
        self.displayer(self)
        self.alert('')

    @view.capture(clear_output=False)
    def _load_handler(self, *_):

        if self.loader is None:
            return

        self.alert('Loading...')
        self.lock(True)
        self.model.update(data=self.loader(self._source_folder.value, filename_fields=self.model.filename_fields))
        self._topic_ids.value = []
        self._topic_ids.options = [("Topic #" + str(i), i) for i in range(0, self.model.num_topics)]

        # self._displayer()

        self.lock(False)
        self.alert('')

    def _relayout_handler(self, *_):
        if self.network:
            self.network.relayout()

    def set_layout(self):
        if not self.network:
            return
        self.network.set_layout(
            name=self.network_layout,
            animate=self.animate,
            **DEFAULT_LAYOUT_ARGUMENTS.get(self.network_layout, {}),
            # nodeSpacing=self.node_spacing,
            # edgeLengthVal=self.edge_length_val,
            # padding=self.padding
        )

    def _layout_handler(self, *_):
        self.set_layout()

    def lock(self, value: bool = True) -> None:
        self._source_folder.disabled = value
        self._topic_ids.disabled = value
        self._button.disabled = value
        if value:
            self._source_folder.unobserve(self._load_handler, names='value')
        else:
            self._source_folder.observe(self._load_handler, names='value')

    def alert(self, msg: str = '&nbsp;') -> None:
        self._label.value = msg or '&nbsp;'

    def setup(
        self,
        folders: str,
        loader: Callable[[str], topic_modelling.InferredTopicsData],
        displayer: Callable[["GUI"], None],
    ) -> "GUI":
        self.loader = loader
        self.displayer = displayer
        self._source_folder.options = {os.path.split(folder)[1]: folder for folder in folders}
        self._source_folder.value = None
        self._source_folder.observe(self._load_handler, names='value')
        self._network_layout.observe(self._layout_handler, names='value')
        self._relayout.on_click(self._relayout_handler)
        self._button.on_click(self._displayer)
        return self

    def layout(self):
        return widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                widgets.HTML("<b>Model</b>"),
                                self._source_folder,
                                widgets.HTML("<b>Output</b>"),
                                self._output_format,
                                widgets.HTML("<b>Top tokens</b>"),
                                self._top_count,
                            ]
                        ),
                        widgets.VBox(
                            [
                                widgets.HTML("<b>Topics</b>"),
                                self._topic_ids,
                            ]
                        ),
                        widgets.VBox(
                            [
                                widgets.HTML("<b>Layout</b>"),
                                self._network_layout,
                                self._relayout,
                                self._button,
                            ]
                        ),
                        widgets.VBox(
                            [
                                self._label,
                                self._animate,
                                self._curve_style,
                                # elf._node_spacing,
                                # self._edge_length_val,
                                # self._padding,
                            ]
                        ),
                    ]
                ),
                view,
            ]
        )

    @property
    def source_folder(self) -> str:
        return self._source_folder.value

    @property
    def topics_ids(self) -> List[int]:
        return self._topic_ids.value

    @property
    def output_format(self) -> List[int]:
        return self._output_format.value

    @property
    def top_count(self) -> List[int]:
        return self._top_count.value

    @property
    def network_layout(self) -> List[int]:
        return self._network_layout.value

    @property
    def curve_style(self) -> str:
        return self._curve_style.value

    @property
    def animate(self) -> bool:
        return self._animate.value

    # @property
    # def node_spacing(self) -> int:
    #     return self._node_spacing.value

    # @property
    # def edge_length_val(self) -> int:
    #     return self._edge_length_val.value

    # @property
    # def padding(self) -> int:
    #     return self._padding.value


def create_gui(data_folder: str):
    gui = GUI(model=ViewModel(filename_fields=['year:_:1', 'sequence_id:_:2'])).setup(
        folders=find_inferred_models(data_folder),
        loader=default_loader,
        displayer=default_displayer,
    )
    return gui
