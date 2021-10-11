import io

import ipycytoscape
import pandas as pd
import pytest
from penelope.notebook.topic_modelling import topics_token_network_gui as ttn_gui
from penelope.topic_modelling import InferredTopicsData

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

    w = ttn_gui.create_network(topics_tokens)

    assert w is not None

    assert len(w.graph.nodes) == 3
    assert len([node for node in w.graph.nodes if node.data['node_type'] == 'topic']) == 1
    assert len([node for node in w.graph.nodes if node.data['node_type'] == 'token']) == 2


def test_create_network2():

    topics_tokens_str = ';topic;token;weight;topic_id;position\n0;Topic #0;och;0.04476533457636833;0;1\n1;Topic #0;valv;0.02178177796304226;0;2\n2;Topic #0;i;0.02060970477759838;0;3\n3;Topic #1;och;0.02447959966957569;1;1\n4;Topic #1;som;0.02074943669140339;1;2\n5;Topic #1;av;0.020712170749902725;1;3\n6;Topic #2;som;0.02533087506890297;2;1\n7;Topic #2;en;0.023466676473617557;2;2\n8;Topic #2;är;0.022432737052440643;2;3\n9;Topic #3;de;0.02712307684123516;3;1\n10;Topic #3;är;0.023767853155732155;3;2\n11;Topic #3;i;0.019748181104660038;3;3\n'
    topics_tokens = pd.read_csv(io.StringIO(topics_tokens_str), sep=';', index_col=0)
    network = ttn_gui.create_network(topics_tokens)
    assert network is not None
    source_network_data = ttn_gui.to_dict(topics_tokens=topics_tokens)
    w = ipycytoscape.CytoscapeWidget(cytoscape_layout={'name': 'euler'})
    w.graph.add_graph_from_json(source_network_data)


data_str = """\ttopic\ttoken\tweight\ttopic_id\tposition
0\tTopic #4\tfartyg\t0.03834409900760612\t4\t1
1\tTopic #4\tjärnväg\t0.022307209037066858\t4\t2
2\tTopic #4\tår\t0.019546560778788268\t4\t3
3\tTopic #4\thamn\t0.015181735560882481\t4\t4
4\tTopic #4\tton\t0.013654412706545408\t4\t5
5\tTopic #4\ttrafik\t0.011393926085869213\t4\t6
6\tTopic #4\tkostnad\t0.00953966830744082\t4\t7
7\tTopic #4\tsjöman\t0.009518929898076815\t4\t8
8\tTopic #4\ttransport\t0.009374980938961979\t4\t9
9\tTopic #4\tgods\t0.007868396493988909\t4\t10
10\tTopic #4\tkr\t0.007314558973326745\t4\t11
11\tTopic #4\tdel\t0.00525779672699104\t4\t12
12\tTopic #4\tsjöfart\t0.004891824797038067\t4\t13
13\tTopic #4\tgöteborg\t0.004603926878808395\t4\t14
14\tTopic #4\tbefälhavare\t0.00444899876179497\t4\t15
15\tTopic #4\tlinje\t0.004391663159435671\t4\t16
16\tTopic #4\tutredning\t0.003991533849353754\t4\t17
17\tTopic #4\tredare\t0.003987874130054225\t4\t18
18\tTopic #4\tstat\t0.0035169902468480674\t4\t19
19\tTopic #4\tsjöfartsverk\t0.003409638480728528\t4\t20
20\tTopic #4\tkm\t0.003340103814037464\t4\t21
21\tTopic #4\tstockholm\t0.0032705691473463986\t4\t22
22\tTopic #4\tresa\t0.003259589989447809\t4\t23
23\tTopic #4\tkrona\t0.0031827358841576853\t4\t24
24\tTopic #4\tbåt\t0.003166877100526389\t4\t25
25\tTopic #4\tsj\t0.0029863309484162567\t4\t26
26\tTopic #4\tantal\t0.0029655925390522546\t4\t27
27\tTopic #4\tväg\t0.00289605787236119\t4\t28
28\tTopic #4\trederi\t0.002889958340195307\t4\t29
29\tTopic #4\tmilj\t0.002872879650130835\t4\t30
30\tTopic #4\tkanal\t0.002650856679292698\t4\t31
31\tTopic #4\ttaxa\t0.0026240187377628136\t4\t32
32\tTopic #4\tfart\t0.0026057201412651647\t4\t33
33\tTopic #4\tfarled\t0.0024581114628507987\t4\t34
34\tTopic #4\tavgift\t0.002428833708454561\t4\t35
35\tTopic #4\tvagn\t0.002372718012528439\t4\t36
36\tTopic #4\tland\t0.002309282878003257\t4\t37
37\tTopic #4\tlastbil\t0.0023019634394041977\t4\t38
38\tTopic #4\tmeter\t0.002223889427680897\t4\t39
39\tTopic #4\tdrift\t0.0022214496148145442\t4\t40
40\tTopic #4\ttonnage\t0.0021153177551281816\t4\t41
41\tTopic #4\tbandel\t0.0020787205621328846\t4\t42
42\tTopic #4\tstorlek\t0.0020274844919394683\t4\t43
43\tTopic #4\tgodstrafik\t0.0019982067375432307\t4\t44
44\tTopic #4\thänsyn\t0.001954290105948874\t4\t45
45\tTopic #4\tuppgift\t0.001899394316455928\t4\t46
46\tTopic #4\ttransportmedel\t0.0018688966556265132\t4\t47
47\tTopic #4\tbesättning\t0.0018603573105942773\t4\t48
48\tTopic #4\tsverige\t0.0018201003982994503\t4\t49
49\tTopic #4\tkaj\t0.0017871629246036828\t4\t50
50\tTopic #5\telev\t0.05281984459931966\t5\t1
51\tTopic #5\tskola\t0.03184916172609606\t5\t2
52\tTopic #5\tlärare\t0.022911214909610737\t5\t3
53\tTopic #5\tundervisning\t0.01990770254662488\t5\t4
54\tTopic #5\tårskurs\t0.01271982814073313\t5\t5
55\tTopic #5\tämne\t0.012080889999863443\t5\t6
56\tTopic #5\tgrundskola\t0.010396220706051802\t5\t7
57\tTopic #5\tarbete\t0.007806688510828528\t5\t8
58\tTopic #5\tlinje\t0.007356053849045294\t5\t9
59\tTopic #5\tgymnasium\t0.00720296743284142\t5\t10
60\tTopic #5\tantal\t0.006581997181197538\t5\t11
61\tTopic #5\tklass\t0.006408786729060761\t5\t12
62\tTopic #5\tskall\t0.006262168752978178\t5\t13
63\tTopic #5\tläsår\t0.005971088947520108\t5\t14
64\tTopic #5\tmoment\t0.0054780500671639705\t5\t15
65\tTopic #5\tuppgift\t0.0050935372377317045\t5\t16
66\tTopic #5\tåk\t0.0048534862376749255\t5\t17
67\tTopic #5\tgymnasieskola\t0.004640746429241375\t5\t18
68\tTopic #5\tfackskola\t0.00431013726748653\t5\t19
69\tTopic #5\thögstadium\t0.004274920204603947\t5\t20
70\tTopic #5\tdel\t0.0042145480968052386\t5\t21
71\tTopic #5\trektor\t0.004022651039873621\t5\t22
72\tTopic #5\tläroplan\t0.003803442791318778\t5\t23
73\tTopic #5\tläromedel\t0.003646762797269743\t5\t24
74\tTopic #5\tspråk\t0.0036151393122323227\t5\t25
75\tTopic #5\tgrupp\t0.003379400605589739\t5\t26
76\tTopic #5\tkunskap\t0.0030473540126968288\t5\t27
77\tTopic #5\tsamband\t0.0030164492432284413\t5\t28
78\tTopic #5\tskolledare\t0.002888517871940697\t5\t29
79\tTopic #5\tlektion\t0.002727525584477469\t5\t30
80\tTopic #5\tfår\t0.002672184485661984\t5\t31
81\tTopic #5\tskolöverstyrelsen\t0.0026700283389548872\t5\t32
82\tTopic #5\tskolstyrelse\t0.002646310725176822\t5\t33
83\tTopic #5\ttid\t0.002603906506603918\t5\t34
84\tTopic #5\tskolform\t0.002603187791034886\t5\t35
85\tTopic #5\tsö\t0.0025471279766503692\t5\t36
86\tTopic #5\thjälpmedel\t0.002546409261081336\t5\t37
87\tTopic #5\tkommun\t0.002476693850885206\t5\t38
88\tTopic #5\ttabell\t0.0024623195395045605\t5\t39
89\tTopic #5\tanvisning\t0.002448663943692948\t5\t40
90\tTopic #5\tavsnitt\t0.0024400393568645602\t5\t41
91\tTopic #5\tmöjlighet\t0.0023882918358942365\t5\t42
92\tTopic #5\tstadion\t0.00237032394666843\t5\t43
93\tTopic #5\tlänsskolnämnd\t0.002337981746061977\t5\t44
94\tTopic #5\tproblem\t0.002337263030492945\t5\t45
95\tTopic #5\tmellanstadium\t0.0023343881682168165\t5\t46
96\tTopic #5\tskolväsen\t0.0023250448658193967\t5\t47
97\tTopic #5\tkrav\t0.002307795692162623\t5\t48
98\tTopic #5\tform\t0.0022366428508284273\t5\t49
99\tTopic #5\tmål\t0.002223705970585847\t5\t50
"""


def test_create_network_5():
    topic_tokens = pd.read_csv(io.StringIO(data_str), sep='\t', index_col=0)
    assert topic_tokens is not None
    w = ttn_gui.create_network(topic_tokens)
    assert w is not None

    w.set_layout(name="cola", animate=False, maxSimulationTime=20000)

    g = ttn_gui.create_networkx(topic_tokens)

    assert g is not None
    assert w is not None
