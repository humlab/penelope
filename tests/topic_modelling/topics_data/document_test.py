import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from penelope.topic_modelling.topics_data import DocumentTopicsCalculator, InferredTopicsData
from penelope.topic_modelling.topics_data.document import (
    compute_topic_proportions,
    filter_by_data_keys,
    filter_by_document_index_keys,
    filter_by_inner_join,
    filter_by_keys,
    filter_by_n_top,
    filter_by_text,
    filter_by_threshold,
    filter_by_topics,
    overload,
)

# pylint: disable=redefined-outer-name


def test_compute_topic_proportions():
    dtw: pd.DataFrame = pd.DataFrame(
        {'document_id': [0, 0, 1, 1, 2, 2], 'topic_id': [0, 1, 0, 2, 1, 2], 'weight': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    )
    document_index: pd.DataFrame = pd.DataFrame({'n_tokens': [10, 20, 30]})

    topic_proportion: pd.DataFrame = compute_topic_proportions(dtw, document_index)

    expected_result: np.NDArray[np.float64] = np.array([0.14, 0.34, 0.52])
    np.testing.assert_array_equal(topic_proportion, expected_result)


def test_compute_topic_proportions_with_last_document_missing_in_dtw():
    dtw: pd.DataFrame = pd.DataFrame(
        {'document_id': [0, 0, 1, 1], 'topic_id': [0, 1, 0, 2], 'weight': [0.1, 0.2, 0.3, 0.4]}
    )
    document_index: pd.DataFrame = pd.DataFrame({'n_tokens': [10, 20]})

    topic_proportion: pd.DataFrame = compute_topic_proportions(dtw, document_index)

    expected_result: np.NDArray[np.float64] = np.array([0.41176471, 0.11764706, 0.47058824])
    np.testing.assert_allclose(topic_proportion, expected_result, atol=0.001)


def test_overload():
    dtw = pd.DataFrame({'document_id': ['doc1', 'doc2'], 'topic1': [0.1, 0.2], 'topic2': [0.2, 0.1]})
    document_index = pd.DataFrame(
        {'document_id': ['doc1', 'doc2'], 'author': ['author1', 'author2'], 'date': ['2021-01-01', '2021-01-02']}
    )

    result = overload(dtw=dtw, document_index=document_index, includes='author,date')
    expected = pd.DataFrame(
        {
            'document_id': ['doc1', 'doc2'],
            'topic1': [0.1, 0.2],
            'topic2': [0.2, 0.1],
            'author': ['author1', 'author2'],
            'date': ['2021-01-01', '2021-01-02'],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_overload_with_ignores():
    dtw = pd.DataFrame({'document_id': ['doc1', 'doc2'], 'topic1': [0.1, 0.2], 'topic2': [0.2, 0.1]})
    document_index = pd.DataFrame(
        {'document_id': ['doc1', 'doc2'], 'author': ['author1', 'author2'], 'date': ['2021-01-01', '2021-01-02']}
    )

    result = overload(dtw=dtw, document_index=document_index, includes='author,date', ignores='date')
    expected = pd.DataFrame(
        {'document_id': ['doc1', 'doc2'], 'topic1': [0.1, 0.2], 'topic2': [0.2, 0.1], 'author': ['author1', 'author2']}
    )
    pd.testing.assert_frame_equal(result, expected)


def test_filter_by_threshold():
    dtw = pd.DataFrame({'weight': [0.5, 0.3, 0.7]}, index=[0, 1, 2])
    threshold = 0.0
    expected_result = dtw.copy()
    assert filter_by_threshold(dtw, threshold).equals(expected_result)

    dtw = pd.DataFrame({'weight': [0.5, 0.3, 0.7]}, index=[0, 1, 2])
    threshold = 0.4
    expected_result = pd.DataFrame({'weight': [0.5, 0.7]}, index=[0, 2])
    assert filter_by_threshold(dtw, threshold).equals(expected_result)

    dtw = pd.DataFrame({'weight': [0.1, 0.2, 0.3]}, index=[0, 1, 2])
    threshold = 0.4
    assert len(filter_by_threshold(dtw, threshold)) == 0


def test_filter_by_data_keys_no_filter():
    dtw = pd.DataFrame({'document_id': ['doc1', 'doc2', 'doc3'], 'topic1': [0.1, 0.2, 0.3], 'topic2': [0.2, 0.1, 0.4]})
    result = filter_by_data_keys(dtw)
    pd.testing.assert_frame_equal(result, dtw)


def test_filter_by_data_keys_single_filter():
    dtw = pd.DataFrame({'document_id': ['doc1', 'doc2', 'doc3'], 'topic1': [0.1, 0.2, 0.3], 'topic2': [0.2, 0.1, 0.4]})
    result = filter_by_data_keys(dtw, document_id='doc1')
    expected = dtw[dtw['document_id'] == 'doc1']
    pd.testing.assert_frame_equal(result, expected)


def test_filter_by_data_keys_multiple_filters():
    dtw = pd.DataFrame({'document_id': ['doc1', 'doc2', 'doc3'], 'topic1': [0.1, 0.2, 0.3], 'topic2': [0.2, 0.1, 0.4]})
    result = filter_by_data_keys(dtw, document_id='doc1', topic1=0.1)
    expected = dtw[(dtw['document_id'] == 'doc1') & (dtw['topic1'] == 0.1)]
    pd.testing.assert_frame_equal(result, expected)


def test_filter_by_document_index_keys_no_filter():
    dtw = pd.DataFrame({'document_id': ['doc1', 'doc2', 'doc3'], 'topic1': [0.1, 0.2, 0.3], 'topic2': [0.2, 0.1, 0.4]})
    document_index = pd.DataFrame(
        {
            'document_id': ['doc1', 'doc2', 'doc3'],
            'author': ['author1', 'author2', 'author3'],
            'date': ['2021-01-01', '2021-01-02', '2021-01-03'],
        }
    )
    result = filter_by_document_index_keys(dtw, document_index=document_index)
    pd.testing.assert_frame_equal(result, dtw)


def test_filter_by_document_index_keys_single_filter():
    dtw = pd.DataFrame({'document_id': ['doc1', 'doc2', 'doc3'], 'topic1': [0.1, 0.2, 0.3], 'topic2': [0.2, 0.1, 0.4]})
    document_index = pd.DataFrame(
        {
            'document_id': ['doc1', 'doc2', 'doc3'],
            'author': ['author1', 'author2', 'author3'],
            'date': ['2021-01-01', '2021-01-02', '2021-01-03'],
        }
    )
    result = filter_by_document_index_keys(dtw, document_index=document_index, author='author1')
    expected = dtw[dtw['document_id'] == 'doc1']
    pd.testing.assert_frame_equal(result, expected)


def test_filter_by_document_index_keys_multiple_filters():
    dtw = pd.DataFrame({'document_id': ['doc1', 'doc2', 'doc3'], 'topic1': [0.1, 0.2, 0.3], 'topic2': [0.2, 0.1, 0.4]})
    document_index = pd.DataFrame(
        {
            'document_id': ['doc1', 'doc2', 'doc3'],
            'author': ['author1', 'author2', 'author3'],
            'date': ['2021-01-01', '2021-01-02', '2021-01-03'],
        }
    )
    result = filter_by_document_index_keys(dtw, document_index=document_index, author='author1', date='2021-01-01')
    expected = dtw[dtw['document_id'] == 'doc1']
    pd.testing.assert_frame_equal(result, expected)


def test_filter_by_keys_no_filter():
    dtw = pd.DataFrame({'document_id': ['doc1', 'doc2', 'doc3'], 'topic1': [0.1, 0.2, 0.3], 'topic2': [0.2, 0.1, 0.4]})
    document_index = pd.DataFrame(
        {
            'document_id': ['doc1', 'doc2', 'doc3'],
            'author': ['author1', 'author2', 'author3'],
            'date': ['2021-01-01', '2021-01-02', '2021-01-03'],
        }
    )
    result = filter_by_keys(dtw, document_index=document_index)
    pd.testing.assert_frame_equal(result, dtw)


def test_filter_by_keys_single_filter():
    dtw = pd.DataFrame({'document_id': ['doc1', 'doc2', 'doc3'], 'topic1': [0.1, 0.2, 0.3], 'topic2': [0.2, 0.1, 0.4]})
    document_index = pd.DataFrame(
        {
            'document_id': ['doc1', 'doc2', 'doc3'],
            'author': ['author1', 'author2', 'author3'],
            'date': ['2021-01-01', '2021-01-02', '2021-01-03'],
        }
    )
    result = filter_by_keys(dtw, document_index=document_index, author='author1')
    expected = dtw[dtw['document_id'] == 'doc1']
    pd.testing.assert_frame_equal(result, expected)


def test_filter_by_keys_multiple_filters():
    dtw = pd.DataFrame({'document_id': ['doc1', 'doc2', 'doc3'], 'topic1': [0.1, 0.2, 0.3], 'topic2': [0.2, 0.1, 0.4]})
    document_index = pd.DataFrame(
        {
            'document_id': ['doc1', 'doc2', 'doc3'],
            'author': ['author1', 'author2', 'author3'],
            'date': ['2021-01-01', '2021-01-02', '2021-01-03'],
        }
    )
    result = filter_by_keys(dtw, document_index=document_index, author='author1', date='2021-01-01')
    expected = dtw[dtw['document_id'] == 'doc1']
    pd.testing.assert_frame_equal(result, expected)


def test_filter_by_topics_no_topics():
    dtw = pd.DataFrame({'document_id': ['doc1', 'doc2', 'doc3'], 'topic_id': [1, 2, 3]})
    result = filter_by_topics(dtw, [])
    assert len(result) == 0


def test_filter_by_topics_single_topic():
    dtw = pd.DataFrame({'document_id': ['doc1', 'doc2', 'doc3'], 'topic_id': [1, 2, 3]})
    result = filter_by_topics(dtw, [1])
    expected = dtw[dtw['topic_id'] == 1]
    pd.testing.assert_frame_equal(result, expected)


def test_filter_by_topics_multiple_topics():
    dtw = pd.DataFrame({'document_id': ['doc1', 'doc2', 'doc3'], 'topic_id': [1, 2, 3]})
    result = filter_by_topics(dtw, [1, 2])
    expected = dtw[dtw['topic_id'].isin([1, 2])]
    pd.testing.assert_frame_equal(result, expected)


def test_filter_by_text_short_search_text():
    dtw = pd.DataFrame({'document_id': ['doc1', 'doc2', 'doc3'], 'topic_id': [0, 1, 2]})
    topic_token_overview = pd.DataFrame({'topic_id': [0, 1, 2], 'tokens': ['token1', 'token2', 'token3']})
    result = filter_by_text(dtw, topic_token_overview, 'to', 2)
    pd.testing.assert_frame_equal(result, dtw)


def test_filter_by_text_single_topic():
    dtw = pd.DataFrame({'document_id': [0, 1, 2], 'topic_id': [0, 1, 2]})
    topic_token_overview = pd.DataFrame({'topic_id': [0, 1, 2], 'tokens': ['token1', 'token2', 'token3']})
    result = filter_by_text(dtw, topic_token_overview, 'token1', 2)
    expected = dtw[dtw['topic_id'] == 0]
    pd.testing.assert_frame_equal(result, expected)


def test_filter_by_text_n_topics():
    dtw = pd.DataFrame({'document_id': [0, 1, 2], 'topic_id': [0, 1, 2]})
    topic_token_overview = pd.DataFrame({'topic_id': [0, 1, 2], 'tokens': ['kalle token1', 'token2 kula', 'token3']})
    result = filter_by_text(dtw, topic_token_overview, 'token', 2)
    expected = dtw
    pd.testing.assert_frame_equal(result, expected)

    topic_token_overview = pd.DataFrame({'topic_id': [0, 1, 2], 'tokens': ['kalle token1', 'token2 kula', 'token3']})
    result = filter_by_text(dtw, topic_token_overview, 'token', 1)
    pd.testing.assert_frame_equal(result, dtw[dtw.topic_id.isin([1, 2])])


def test_filter_by_inner_join_with_series():
    dtw = pd.DataFrame({'document_id': [0, 1, 2], 'topic_id': [1, 2, 3]}, index=[1, 2, 3])
    other = pd.Series([1, 2], index=[1, 2])
    result = filter_by_inner_join(dtw, other)
    expected = dtw.loc[[1, 2]]
    pd.testing.assert_frame_equal(result, expected)


def test_filter_by_inner_join_with_dataframe():
    dtw = pd.DataFrame({'document_id': [0, 1, 2], 'topic_id': [1, 2, 3]}, index=[1, 2, 3])
    other = pd.DataFrame(index=[1, 2])
    result = filter_by_inner_join(dtw, other)
    expected = dtw.loc[[1, 2]]
    pd.testing.assert_frame_equal(result, expected)


def test_filter_by_inner_join_with_sequence():
    dtw = pd.DataFrame({'document_id': [0, 1, 2], 'topic_id': [1, 2, 3]}, index=[1, 2, 3])
    other = pd.DataFrame(index=[1, 2])
    result = filter_by_inner_join(dtw, other)
    expected = dtw.loc[[1, 2]]
    pd.testing.assert_frame_equal(result, expected)


def test_filter_by_inner_join_with_left_on():
    dtw = pd.DataFrame({'document_id': [0, 1, 2], 'topic_id': [1, 2, 3]})
    other = pd.DataFrame(index=[1, 2])
    result = filter_by_inner_join(dtw, other, left_index=False, left_on='topic_id')
    expected = dtw[dtw['topic_id'].isin([1, 2])]
    pd.testing.assert_frame_equal(result, expected)


def test_filter_by_n_top_no_filter():
    dtw = pd.DataFrame({'document_id': [0, 1, 2], 'weight': [0.1, 0.2, 0.3]})
    result = filter_by_n_top(dtw, 3)
    pd.testing.assert_frame_equal(result, dtw.set_index('document_id').sort_values('weight', ascending=False))


def test_filter_by_n_top_single_filter():
    dtw = pd.DataFrame({'document_id': [0, 1, 2], 'weight': [0.1, 0.2, 0.3]})
    result = filter_by_n_top(dtw, 2)
    expected = dtw.set_index('document_id').sort_values('weight', ascending=False).head(2)
    pd.testing.assert_frame_equal(result, expected)


def test_filter_by_n_top_multiple_filters():
    dtw = pd.DataFrame({'document_id': [0, 1, 2, 3, 4], 'weight': [0.1, 0.2, 0.3, 0.4, 0.5]})
    result = filter_by_n_top(dtw, 3)
    expected = dtw.set_index('document_id').sort_values('weight', ascending=False).head(3)
    pd.testing.assert_frame_equal(result, expected)


@pytest.fixture
def inferred_topics() -> InferredTopicsData:
    data: InferredTopicsData = InferredTopicsData(
        document_topic_weights=pd.DataFrame(
            {
                'document_id': [0, 1, 1, 0, 1, 2],
                'topic_id': [0, 0, 1, 0, 1, 2],
                'weight': [1.0, 0.5, 0.5, 0.3333, 0.3333, 0.3333],
            }
        ),
        document_index=pd.DataFrame({'document_id': [0, 1, 2]}),
        topic_token_overview=pd.DataFrame({'topic_id': [0, 1, 2], '': ['token1', 'token2', 'token3']}),
        dictionary=pd.DataFrame({'token_id': [0, 1, 2], 'token': ['token1', 'token2', 'token3']}),
        topic_token_weights=pd.DataFrame(
            {
                'topic_id': [0, 0, 0, 1, 1, 2],
                'token_id': [0, 1, 2, 0, 1, 2],
                'weight': [0.8, 0.15, 0.05, 0.5, 0.5, 1.0],
            }
        ),
        topic_diagnostics=None,
        token_diagnostics=None,
    )

    return data


@pytest.fixture
def calculator(inferred_topics: InferredTopicsData) -> DocumentTopicsCalculator:
    return DocumentTopicsCalculator(inferred_topics)


def test_reset_no_data(inferred_topics: InferredTopicsData):
    calculator = DocumentTopicsCalculator(inferred_topics)
    calculator.data = pd.DataFrame(
        {'document_id': ['doc4', 'doc5', 'doc6'], 'topic_id': [4, 5, 6], 'weight': [0.4, 0.5, 0.6]}
    )

    reset: DocumentTopicsCalculator = calculator.reset()
    assert reset is calculator
    pd.testing.assert_frame_equal(reset.data, inferred_topics.document_topic_weights)


def test_reset_with_data(inferred_topics: InferredTopicsData):
    calculator = DocumentTopicsCalculator(inferred_topics)
    new_data = pd.DataFrame({'document_id': ['doc4', 'doc5', 'doc6'], 'topic_id': [4, 5, 6], 'weight': [0.4, 0.5, 0.6]})

    reset: DocumentTopicsCalculator = calculator.reset(new_data)
    assert reset is calculator
    pd.testing.assert_frame_equal(reset.data, new_data)
