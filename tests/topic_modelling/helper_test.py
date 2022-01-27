import operator
from typing import List

import pandas as pd

from penelope.notebook.topic_modelling import TopicModelContainer
from penelope.topic_modelling.topics_data import document as helper


def test_weights_reducer(state: TopicModelContainer):

    calculator: helper.DocumentTopicsCalculator = helper.DocumentTopicsCalculator(state.inferred_topics)
    default_columns: List[str] = state.inferred_topics.document_topic_weights.columns.tolist()

    assert calculator.data is state.inferred_topics.document_topic_weights
    assert calculator.value is state.inferred_topics.document_topic_weights
    assert calculator.copy().value is not state.inferred_topics.document_topic_weights
    assert calculator.reset().value is state.inferred_topics.document_topic_weights

    assert calculator.reset().filter_by_keys(year=2019).value.year.unique().tolist() == [2019]
    assert calculator.reset().filter_by_keys(year=[2019]).value.year.unique().tolist() == [2019]
    assert calculator.reset().filter_by_keys(year=[0]).value.year.unique().tolist() == []
    assert len(calculator.reset().filter_by_keys(year=[2019], document_id=[0, 1]).value) == 8

    assert len(calculator.reset().threshold(0).value) == 20
    assert len(calculator.reset().threshold(1.0).value) == 0
    assert len(calculator.reset().threshold(0.9).value) == 5

    assert calculator.reset().filter_by_data_keys(year=[2019]).value.year.unique().tolist() == [2019]
    assert len(calculator.reset().filter_by_data_keys(document_id=0).value) == 4
    assert len(calculator.reset().filter_by_data_keys(topic_id=0).value) == 5

    overloaded: pd.DataFrame = calculator.reset().overload(includes='document_name').value
    assert set(overloaded.columns) == {'year', 'topic_id', 'document_name', 'weight', 'document_id'}
    assert (overloaded[default_columns] == state.inferred_topics.document_topic_weights[default_columns]).all().all()

    overloaded: pd.DataFrame = calculator.reset().overload().value
    assert set(overloaded.columns) - set(default_columns) == set(state.inferred_topics.document_index.columns) - set(
        default_columns
    )

    data: pd.DataFrame = calculator.reset().filter_by_document_keys(title='Nocturne').value
    assert len(data) == 4
    assert data.document_id.unique().tolist() == [1]

    data: pd.DataFrame = calculator.reset().filter_by_document_keys(n_tokens=(operator.ge, 100)).value
    assert data.document_id.unique().tolist() == [2]

    data: pd.DataFrame = calculator.reset().filter_by_document_keys(n_tokens=(operator.lt, 60)).value
    assert data.document_id.unique().tolist() == [1, 3, 4]

    data: pd.DataFrame = calculator.reset().filter_by_topics(topic_ids=[0, 2]).value
    assert set(data.topic_id.unique()) == {0, 2}

    data: pd.DataFrame = calculator.reset().filter_by_text(search_text='APA', n_top=2).value
    assert len(data) == 0

    data: pd.DataFrame = calculator.reset().filter_by_text(search_text='och', n_top=2).value
    assert len(data) == 10
    assert set(data.topic_id.unique()) == {0, 1}

    data: pd.DataFrame = (
        calculator.reset().threshold(threshold=0.1).filter_by_keys(year=[2019, 2020], n_tokens=('ge', 60)).value
    )
    data = data.assign(weight=data.weight.round(3))
    assert data.to_csv(sep=';') == ';document_id;topic_id;weight;year\n0;0;0;0.992;2019\n10;2;2;0.997;2019\n'


def test_filter_by_keys(state: TopicModelContainer):

    calculator: helper.DocumentTopicsCalculator = helper.DocumentTopicsCalculator(state.inferred_topics)
    years = (2020, 2020)
    threshold: float = 0.01
    document_topics: pd.DataFrame = (
        calculator.filter_by_document_keys(year=years)
        .filter_by_text(search_text='och', n_top=100)
        .threshold(threshold)
        .value
    )
    assert document_topics is not None
    assert set(document_topics.year) == {2020}
    assert (document_topics.weight >= threshold).all()
