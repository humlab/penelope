import io
import types

import ipycytoscape
import pandas as pd
import pytest
from penelope.notebook.topic_modelling import topics_token_network_gui as ttn_gui
from penelope.topic_modelling.container import InferredTopicsData

INFERRED_TOPICS_DATA_FOLDER = './tests/test_data/tranströmer_inferred_model'

# pylint: disable=protected-access, redefined-outer-name


def load_inferred_topics_data() -> InferredTopicsData:
    inferred_data: InferredTopicsData = InferredTopicsData.load(
        folder=INFERRED_TOPICS_DATA_FOLDER, filename_fields="year:_:1"
    )
    return inferred_data


@pytest.fixture
def inferred_topics_data() -> InferredTopicsData:
    return load_inferred_topics_data()


def test_find_inferred_models():
    folders = ttn_gui.find_inferred_models(INFERRED_TOPICS_DATA_FOLDER)
    assert folders == [INFERRED_TOPICS_DATA_FOLDER]


def test_view_model(inferred_topics_data: InferredTopicsData):

    assert inferred_topics_data is not None

    model = ttn_gui.ViewModel(filename_fields="year:_:1")

    assert model is not None

    model.update(inferred_topics_data)

    assert model._topics_data is inferred_topics_data
    assert model.top_topic_tokens is not None
    assert len(model.get_topics_tokens([0, 1, 2, 3], 1)) == 4
    assert len(model.get_topics_tokens([0, 1, 2, 3], 3)) == 12

    assert model.get_topics_tokens([0, 1], 1).columns.tolist() == ['topic', 'token', 'weight', 'topic_id', 'position']
    assert list(model.get_topics_tokens([0, 1], 1).topic_id.unique()) == [0, 1]


def test_to_dict(inferred_topics_data: InferredTopicsData):

    model = ttn_gui.ViewModel(filename_fields="year:_:1").update(inferred_topics_data)

    topics_tokens = model.get_topics_tokens([0], 3)
    graph = ttn_gui.to_dict(topics_tokens)

    assert isinstance(graph, dict) and 'nodes' in graph and 'edges' in graph

    nodes = graph['nodes']
    edges = graph['edges']
    assert len(nodes) == 4
    assert len(edges) == 3
    assert {x['data']['id'] for x in nodes} == {'valv', 'i', 'Topic #0', 'och'}
    assert {x['data']['id'] for x in edges} == {'0_valv', '0_i', '0_och'}


def test_create_network(inferred_topics_data: InferredTopicsData):

    n_top_count = 2
    topics_ids = [1]
    model = ttn_gui.ViewModel(filename_fields="year:_:1")
    model.update(inferred_topics_data)
    topics_tokens = model.get_topics_tokens(topics_ids, n_top_count)

    opts = types.SimpleNamespace(network_layout="cola")
    w = ttn_gui.create_network(topics_tokens, opts)

    assert w is not None

    assert len(w.graph.nodes) == 3
    assert len([node for node in w.graph.nodes if node.data['node_type'] == 'topic']) == 1
    assert len([node for node in w.graph.nodes if node.data['node_type'] == 'token']) == 2


def test_create_network2():

    topics_tokens_str = ';topic;token;weight;topic_id;position\n0;Topic #0;och;0.04476533457636833;0;1\n1;Topic #0;valv;0.02178177796304226;0;2\n2;Topic #0;i;0.02060970477759838;0;3\n3;Topic #1;och;0.02447959966957569;1;1\n4;Topic #1;som;0.02074943669140339;1;2\n5;Topic #1;av;0.020712170749902725;1;3\n6;Topic #2;som;0.02533087506890297;2;1\n7;Topic #2;en;0.023466676473617557;2;2\n8;Topic #2;är;0.022432737052440643;2;3\n9;Topic #3;de;0.02712307684123516;3;1\n10;Topic #3;är;0.023767853155732155;3;2\n11;Topic #3;i;0.019748181104660038;3;3\n'
    topics_tokens = pd.read_csv(io.StringIO(topics_tokens_str), sep=';', index_col=0)
    opts = types.SimpleNamespace(network_layout="cola")
    network = ttn_gui.create_network(topics_tokens, opts)
    assert network is not None
    source_network_data = ttn_gui.to_dict(topics_tokens=topics_tokens)
    w = ipycytoscape.CytoscapeWidget(cytoscape_layout={'name': 'euler'})
    w.graph.add_graph_from_json(source_network_data)
