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

    assert model.get_topics_tokens([0, 1], 1).columns.tolist() == ['topic_id', 'token', 'weight']
    assert list(model.get_topics_tokens([0, 1], 1).topic_id.unique()) == [0, 1]


def test_to_dict():

    topics_tokens_str = """;topic_id;token;weight\n0;1;och;0.024\n1;1;som;0.020"""
    topics_tokens = pd.read_csv(io.StringIO(topics_tokens_str), sep=';', index_col=0)
    data = ttn_gui.to_dict(topics_tokens)

    assert isinstance(data, dict)


def test_create_network(inferred_topics_data: InferredTopicsData):

    n_top_count = 2
    topics_ids = [1]
    model = ttn_gui.ViewModel(filename_fields="year:_:1")
    model.update(inferred_topics_data)
    topics_tokens = model.get_topics_tokens(topics_ids, n_top_count)

    opts = types.SimpleNamespace(network_layout="cola")
    w = ttn_gui.create_network(topics_tokens, opts)

    assert w is not None

    assert len(w.graph.nodes) == 4
    assert len([node for node in w.graph.nodes if node.classes == 'source_class']) == 1
    assert len([node for node in w.graph.nodes if node.classes == 'target_class']) == 3

    assert len([node.classes == 'source_class' and node.data['id'] == "1" for node in w.graph.nodes]) == 1

    # [
    #     ipycytoscape.Node(classes='source_class', data={'id': '1', 'label': '1'}, position={}),
    #     ipycytoscape.Node(classes='target_class', data={'id': 'och', 'label': 'och'}, position={}),
    #     ipycytoscape.Node(classes='target_class', data={'id': 'som', 'label': 'som'}, position={}),
    #     ipycytoscape.Node(data={'id': 1}, position={}),
    # ]


def test_create_network2():

    topics_tokens_str = """;topic_id;token;weight
0;2;barn;0.12300785397955755
1;2;förälder;0.02780568784132421
2;2;familj;0.013902092415585314
3;2;år;0.01278234985116442
4;2;behov;0.011685903944124094
5;2;kommun;0.010614257704617923
6;2;förskola;0.010506792478636591
7;2;skola;0.009617761972791006
8;2;hem;0.008996267274283572
9;2;ungdom;0.0084589411443769
10;2;personal;0.008186144801501207
11;2;verksamhet;0.007076923308155408
12;2;dag;0.0066838861529929065
13;2;daghem;0.0065824329676259135
14;2;samhälle;0.006129275406320007
15;2;ålder;0.005966198804656025
16;2;arbete;0.0058489640126763885
17;2;kontakt;0.0056520696825567394
18;2;grupp;0.005506277697658986
19;2;skall;0.005404824512291993
20;2;får;0.005071156258196102
21;2;utredning;0.00491108567683929
22;2;möjlighet;0.004867498382385322
23;2;moder;0.0046668465268817115
24;2;del;0.004642046859347558
25;2;institution;0.004413589316002624
26;2;människa;0.004393298678929226
27;2;tid;0.004386535133238093
28;2;vård;0.004357226435243183
29;2;barnomsorg;0.004333178272785822
30;2;plats;0.004051363868988617
31;2;utveckling;0.003711683574278386
32;2;socialstyrelse;0.0032968527718889003
33;2;erfarenhet;0.003160078847912657
34;2;antal;0.003160078847912657
35;2;situation;0.0030714012488511373
36;2;vårdnadshavare;0.0030488560965473603
37;2;miljö;0.002902312606572814
38;2;form;0.002895549060881681
39;2;omsorg;0.002824907583663181
40;2;utbildning;0.0027738052384412887
41;2;barnavårdsnämnd;0.002674606568304673
42;2;stöd;0.002661830981999199
43;2;tillsyn;0.0026460493753865553
44;2;fritidshem;0.002613734657084476
45;2;samarbete;0.002590437999703907
46;2;problem;0.002531820603714089
47;2;sou;0.0024807182584921953
48;2;hand;0.0024739547128010625
49;2;stockholm;0.0024205978523487914
"""
    topics_tokens = pd.read_csv(io.StringIO(topics_tokens_str), sep=';', index_col=0)
    opts = types.SimpleNamespace(network_layout="cola")
    network = ttn_gui.create_network(topics_tokens, opts)
    assert network is not None
    source_network_data = ttn_gui.to_dict(topics_tokens=topics_tokens)
    w = ipycytoscape.CytoscapeWidget(cytoscape_layout={'name': 'euler'})
    w.graph.add_graph_from_json(source_network_data)
