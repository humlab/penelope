import operator
from typing import List

import pandas as pd
from penelope.notebook.topic_modelling import TopicModelContainer
from penelope.topic_modelling import helper


def test_weights_reducer(state: TopicModelContainer):

    reducer: helper.FilterDocumentTopicWeights = helper.FilterDocumentTopicWeights(state.inferred_topics)
    default_columns: List[str] = state.inferred_topics.document_topic_weights.columns.tolist()

    assert reducer.data is state.inferred_topics.document_topic_weights
    assert reducer.value is state.inferred_topics.document_topic_weights
    assert reducer.copy().value is not state.inferred_topics.document_topic_weights
    assert reducer.reset().value is state.inferred_topics.document_topic_weights

    assert reducer.reset().filter_by_keys(key_values={'year': 2019}).value.year.unique().tolist() == [2019]
    assert reducer.reset().filter_by_keys(key_values={'year': [2019]}).value.year.unique().tolist() == [2019]
    assert reducer.reset().filter_by_keys(key_values={'year': [0]}).value.year.unique().tolist() == []
    assert len(reducer.reset().filter_by_keys(key_values={'year': [2019], 'document_id': [0, 1]}).value) == 8

    assert len(reducer.reset().threshold(0).value) == 20
    assert len(reducer.reset().threshold(1.0).value) == 0
    assert len(reducer.reset().threshold(0.9).value) == 5

    assert reducer.reset().filter_by_data_keys(key_values={'year': [2019]}).value.year.unique().tolist() == [2019]
    assert len(reducer.reset().filter_by_data_keys(key_values={'document_id': 0}).value) == 4
    assert len(reducer.reset().filter_by_data_keys(key_values={'topic_id': 0}).value) == 5

    overloaded: pd.DataFrame = reducer.reset().overload(includes='document_name').value
    assert set(overloaded.columns) == {'year', 'topic_id', 'document_name', 'weight', 'document_id'}
    assert (overloaded[default_columns] == state.inferred_topics.document_topic_weights[default_columns]).all().all()

    overloaded: pd.DataFrame = reducer.reset().overload().value
    assert set(overloaded.columns) - set(default_columns) == set(state.inferred_topics.document_index.columns) - set(
        default_columns
    )

    data: pd.DataFrame = reducer.reset().filter_by_document_keys(key_values={'title': 'Nocturne'}).value
    assert len(data) == 4
    assert data.document_id.unique().tolist() == [1]

    data: pd.DataFrame = reducer.reset().filter_by_document_keys(key_values={'n_terms': (operator.ge, 100)}).value
    assert data.document_id.unique().tolist() == [2]

    data: pd.DataFrame = reducer.reset().filter_by_document_keys(key_values={'n_terms': (operator.lt, 60)}).value
    assert data.document_id.unique().tolist() == [1, 3, 4]

    data: pd.DataFrame = reducer.reset().filter_by_topics(topic_ids=[0, 2]).value
    assert set(data.topic_id.unique()) == {0, 2}

    data: pd.DataFrame = reducer.reset().filter_by_text(search_text='APA', n_top=2).value
    assert len(data) == 0

    data: pd.DataFrame = reducer.reset().filter_by_text(search_text='och', n_top=2).value
    assert len(data) == 10
    assert set(data.topic_id.unique()) == {0, 1}

    data: pd.DataFrame = (
        reducer.reset()
        .filter_by(threshold=0.1, key_values=dict(year=[2019, 2020]), document_key_values=dict(n_terms=('ge', 60)))
        .value
    )
    data = data.assign(weight=data.weight.round(3))
    assert data.to_csv(sep=';') == ';document_id;topic_id;weight;year\n0;0;0;0.992;2019\n10;2;2;0.997;2019\n'
