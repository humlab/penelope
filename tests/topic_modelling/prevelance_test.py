import io

import pandas as pd
import pytest
from penelope.notebook.topic_modelling import TopicModelContainer
from penelope.topic_modelling import prevelance
from penelope.topic_modelling.interfaces import InferredTopicsData

# pylint: disable=protected-access, redefined-outer-name


def test_compute_topic_yearly_means(state: TopicModelContainer):

    document_topic_weights: pd.DataFrame = state.inferred_topics.document_topic_weights
    document_index: pd.DataFrame = state.inferred_topics.document_index

    topic_yearly_mean_weights: pd.DataFrame = prevelance.compute_yearly_topic_weights(
        document_topic_weights, document_index
    )
    assert len(topic_yearly_mean_weights) == 8

    assert (
        document_topic_weights.groupby(['year', 'topic_id'])['weight'].mean()
        == topic_yearly_mean_weights.groupby(['year', 'topic_id']).sum().average_weight
    ).all()

    assert topic_yearly_mean_weights is not None

    assert (topic_yearly_mean_weights[topic_yearly_mean_weights.year == 2019].n_documents == 3).all()
    assert (topic_yearly_mean_weights[topic_yearly_mean_weights.year == 2020].n_documents == 2).all()

    topic_yearly_mean_weights: pd.DataFrame = prevelance.compute_yearly_topic_weights(
        document_topic_weights, document_index=document_index, threshold=0.20
    )

    assert topic_yearly_mean_weights is not None


@pytest.fixture
def simple_test_data():

    data_str: str = """document_id;topic_id;weight;year
0;0;0.74;2019
0;1;0.01;2019
0;2;0.01;2019
0;3;0.24;2019
1;0;0.01;2019
1;1;0.01;2019
1;2;0.38;2019
1;3;0.60;2019
2;0;0.21;2019
2;1;0.33;2019
2;3;0.46;2019
3;0;0.01;2020
3;1;0.98;2020
3;2;0.01;2020
4;0;0.50;2020
4;1;0.50;2020
"""
    # 2;2;0.00;2019
    # 3;3;0.00;2020
    # 4;2;0.00;2020
    # 4;3;0.00;2020
    document_index = pd.read_csv(
        io.StringIO(
            """;filename;year;year_serial_id;document_id;document_name;n_terms
a;a.txt;2019;1;0;a;68
b;b.txt;2019;2;1;b;59
c;c.txt;2019;3;2;c;173
d;d.txt;2020;1;3;d;33
e;e.txt;2020;2;4;e;44
"""
        ),
        sep=';',
    )

    dtw: pd.DataFrame = pd.read_csv(io.StringIO(data_str), sep=';')
    return document_index, dtw


def test_n_top_prevelance_over_time(simple_test_data):
    document_index, document_topic_weights = simple_test_data
    yearly_weights: pd.DataFrame = prevelance.compute_yearly_topic_weights(
        document_topic_weights, document_index=document_index, threshold=0.00, n_top_relevance=1
    )
    assert yearly_weights.sort_index().top_n_weight.round(2).tolist() == [0.33, 0.00, 0.0, 0.67, 0.5, 0.5, 0.0, 0.0]


def test_compute_prevelance_parts(simple_test_data):

    document_index, dtw = simple_test_data

    yearly_weights: pd.DataFrame = prevelance._compute_yearly_topic_weights(dtw)

    assert len(yearly_weights) == 8
    assert yearly_weights.columns.tolist() == ['max_weight', 'sum_weight', 'average_weight', 'n_topic_documents']
    assert yearly_weights.sort_index().average_weight.round(2).tolist() == [
        0.32,
        0.12,
        0.2,
        0.43,
        0.26,
        0.74,
        0.01,
        0.0,
    ]

    yearly_weights = prevelance._add_yearly_corpus_document_count(yearly_weights, document_index)
    assert yearly_weights.n_documents.tolist() == [3, 3, 3, 3, 2, 2, 2, 2]

    yearly_weights = prevelance._add_average_yearly_topic_weight_by_all_documents(yearly_weights)
    assert yearly_weights.true_average_weight.round(2).tolist() == [0.32, 0.12, 0.13, 0.43, 0.26, 0.74, 0.0, 0.0]

    yearly_weights: pd.DataFrame = prevelance._add_average_yearly_topic_weight_above_threshold(
        yearly_weights, document_topic_weights=dtw, threshold=0.0
    )
    assert yearly_weights.sort_index().average_weight.round(2).tolist() == [
        0.32,
        0.12,
        0.2,
        0.43,
        0.26,
        0.74,
        0.01,
        0.0,
    ]

    yearly_weights: pd.DataFrame = prevelance._add_average_yearly_topic_weight_above_threshold(
        yearly_weights, document_topic_weights=dtw, threshold=0.20
    )
    assert yearly_weights.sort_index().average_weight.round(2).tolist() == [0.48, 0.33, 0.38, 0.43, 0.5, 0.74, 0.0, 0.0]
    assert len(yearly_weights) == 8
    assert 'average_weight' in yearly_weights.columns.tolist()

    yearly_weights = prevelance._add_top_n_topic_prevelance_weight(yearly_weights, dtw, n_top_relevance=1)
    assert yearly_weights.sort_index().top_n_weight.round(2).tolist() == [0.33, 0.0, 0.0, 0.67, 0.5, 0.5, 0.0, 0.0]
    assert yearly_weights is not None


def test_create_gui(state: TopicModelContainer):

    inferred_topics: InferredTopicsData = state.inferred_topics
    calculator: prevelance.TopicPrevalenceOverTimeCalculator = prevelance.AverageTopicPrevalenceOverTimeCalculator()

    data: pd.DataFrame = calculator.compute(
        inferred_topics=inferred_topics, filters={}, threshold=0, result_threshold=0
    )
    assert data is not None
    assert len(data) == 8
