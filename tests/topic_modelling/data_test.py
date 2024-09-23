import math
import os
import shutil
import uuid

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from penelope import corpus as pc
from penelope import topic_modelling as ntm
from penelope import topic_modelling as tm
from tests.utils import OUTPUT_FOLDER

jj = os.path.join
isfile = os.path.isfile

# pylint: disable=protected-access,redefined-outer-name


@pytest.fixture
def topics_data() -> tm.InferredTopicsData:
    return tm.InferredTopicsData.load(folder='tests/test_data/tranströmer/tranströmer_inferred_model')


def assert_equal(d1: tm.InferredTopicsData, d2: tm.InferredTopicsData) -> bool:
    assert_frame_equal(d1.document_index, d2.document_index, check_index_type=False)
    assert_frame_equal(d1.dictionary, d2.dictionary)
    assert_frame_equal(d1.topic_token_weights, d2.topic_token_weights)
    assert_frame_equal(d1.topic_token_overview, d2.topic_token_overview)
    assert_frame_equal(d1.document_topic_weights, d2.document_topic_weights)


def test_store_and_load_pickle(topics_data: tm.InferredTopicsData):
    folder: str = f"./tests/output/{str(uuid.uuid4())[:6]}"
    os.makedirs(folder, exist_ok=True)
    tm.PickleUtility.store(data=topics_data, target_folder=folder)
    data: tm.InferredTopicsData = tm.PickleUtility.load(folder=folder)
    assert data is not None
    assert_equal(data, topics_data)


def test_explode_pickle(topics_data: tm.InferredTopicsData):
    folder: str = f"./tests/output/{str(uuid.uuid4())[:6]}"
    os.makedirs(folder, exist_ok=True)
    tm.PickleUtility.explode(source=topics_data, target_folder=folder, feather=True)

    for basename in [
        'documents',
        'dictionary',
        'topic_token_weights',
        'topic_token_overview',
        'document_topic_weights',
    ]:
        assert isfile(jj(folder, f'{basename}.zip'))
        assert isfile(jj(folder, f'{basename}.feather'))


@pytest.mark.parametrize('feather', [False, True])
def test_load(topics_data: tm.InferredTopicsData, feather: bool):
    folder: str = f"./tests/output/{str(uuid.uuid4())[:6]}"
    os.makedirs(folder, exist_ok=True)
    tm.PickleUtility.explode(source=topics_data, target_folder=folder, feather=feather)
    data: tm.InferredTopicsData = tm.InferredTopicsData.load(folder=folder, slim=False)
    assert_equal(data, topics_data)


def test_load_topics_data(topics_data: tm.InferredTopicsData):
    assert topics_data is not None
    assert isinstance(topics_data.document_index, pd.DataFrame)
    assert isinstance(topics_data.dictionary, pd.DataFrame)
    assert isinstance(topics_data.topic_token_weights, pd.DataFrame)
    assert isinstance(topics_data.topic_token_overview, pd.DataFrame)
    assert isinstance(topics_data.document_topic_weights, pd.DataFrame)
    assert topics_data.year_period == (2019, 2020)
    assert set(topics_data.topic_ids) == {0, 1, 2, 3}
    assert len(topics_data.document_index) == 5
    assert list(topics_data.topic_token_weights.topic_id.unique()) == [0, 1, 2, 3]
    assert list(topics_data.topic_token_overview.index) == [0, 1, 2, 3]
    assert set(topics_data.document_topic_weights.topic_id.unique()) == {0, 1, 2, 3}


@pytest.mark.parametrize('format', ['zip', 'feather'])
def test_store_inferred_topics_data_as_zipped_files(topics_data: tm.InferredTopicsData, format: str):
    target_folder: str = jj(OUTPUT_FOLDER, f"{str(uuid.uuid1())[:6]}")

    topics_data.store(target_folder, pickled=False, feather=format == 'feather')

    for filename in [
        "dictionary",
        "document_topic_weights",
        "documents",
        "topic_token_overview",
        "topic_token_weights",
    ]:
        assert isfile(jj(target_folder, f'{filename}.{format}'))

    loaded_data: tm.InferredTopicsData = tm.InferredTopicsData.load(folder=target_folder)

    assert set(topics_data.dictionary.columns) == set(loaded_data.dictionary.columns)
    assert set(topics_data.document_index.columns) == set(loaded_data.document_index.columns)
    assert set(topics_data.document_topic_weights.columns) == set(loaded_data.document_topic_weights.columns)
    assert set(topics_data.topic_token_overview.columns) == set(loaded_data.topic_token_overview.columns)
    assert set(topics_data.topic_token_overview.columns) == set(loaded_data.topic_token_overview.columns)

    assert topics_data.dictionary.equals(loaded_data.dictionary)

    pd.testing.assert_frame_equal(
        topics_data.document_index, loaded_data.document_index, check_index_type=False, check_dtype=False
    )

    assert topics_data.topic_token_overview.tokens.tolist() == loaded_data.topic_token_overview.tokens.tolist()
    assert ((loaded_data.topic_token_weights.weight - topics_data.topic_token_weights.weight) < 0.000000005).all()
    assert (loaded_data.topic_token_weights.topic_id == topics_data.topic_token_weights.topic_id).all()
    assert (loaded_data.topic_token_weights.token_id == topics_data.topic_token_weights.token_id).all()
    assert (loaded_data.topic_token_weights.token == topics_data.topic_token_weights.token).all()

    shutil.rmtree(target_folder, ignore_errors=True)


def test_load_token2id(topics_data: tm.InferredTopicsData):
    token2id: pc.Token2Id = tm.InferredTopicsData.load_token2id(
        'tests/test_data/tranströmer/tranströmer_inferred_model'
    )

    assert token2id.id2token == topics_data.id2token


# def test_merge_inferred_topics():

#     inferred_topics: tm.InferredTopicsData = tm.InferredTopicsData.load(
#         folder='tests/test_data/test-corpus.gensim_mallet-lda', slim=True
#     )

#     assert inferred_topics is not None

#     topics: list[int] = [2, 3]

#     inferred_topics.merge(topics)

#     assert not (inferred_topics.topic_token_weights.topic_id == 3).any()
#     assert not (inferred_topics.document_topic_weights.topic_id == 3).any()

#     assert inferred_topics.topic_token_overview.index.to_list() == [0, 1, 2, 3, 4]

#     inferred_topics.compress()

#     assert inferred_topics.topic_token_overview.index.to_list() == [0, 1, 2, 3]


def test_copy_inferred_topics():

    inferred_topics: tm.InferredTopicsData = tm.InferredTopicsData.load(
        folder='tests/test_data/tranströmer/tranströmer_inferred_model', slim=True
    )

    assert inferred_topics is not None

    inferred_topics_copy = inferred_topics.copy()

    assert inferred_topics_copy is not None

    assert_equal(inferred_topics, inferred_topics_copy)


# def test_merge_transtromer_inferred_topics():

#     inferred_topics: tm.InferredTopicsData = tm.InferredTopicsData.load(
#         folder='tests/test_data/tranströmer/tranströmer_inferred_model', slim=True
#     )

#     topics: list[int] = [2, 3]
#     merged_inferred_topics: tm.InferredTopicsData = inferred_topics.copy().merge(topics)

#     """Make sure that topics 2 and 3 are merged into topic 2"""
#     assert (merged_inferred_topics.document_topic_weights.topic_id == 2).any()
#     assert not (merged_inferred_topics.document_topic_weights.topic_id == 3).any()
#     assert not (merged_inferred_topics.document_topic_weights.topic_id == 3).any()

#     assert (merged_inferred_topics.topic_token_weights.topic_id == 2).any()
#     assert not (merged_inferred_topics.topic_token_weights.topic_id == 3).any()
#     assert not (merged_inferred_topics.topic_token_weights.topic_id == 3).any()

#     """Compare total weights of document topics in the merged data to the original data"""

#     for key in ['document_topic_weights', 'topic_token_weights']:

#         data: pd.DataFrame = getattr(inferred_topics, key)
#         merged_data: pd.DataFrame = getattr(merged_inferred_topics, key)

#         topic_weights = data.groupby('topic_id')['weight'].sum()
#         merged_topic_weights: pd.Series[np.float64] = merged_data.groupby('topic_id')['weight'].sum()

#         assert math.isclose(merged_topic_weights[0], topic_weights[0], rel_tol=1e-5)
#         assert math.isclose(merged_topic_weights[1], topic_weights[1], rel_tol=1e-5)
#         assert math.isclose(merged_topic_weights[2], topic_weights[2] + topic_weights[3], rel_tol=1e-5)

#     """Compare total weights per topic in the merged topic token data and to the original data"""

#     assert merged_inferred_topics.topic_token_overview.index.to_list() == [0, 1, 2, 3]

#     merged_inferred_topics.compress()

#     assert merged_inferred_topics.topic_token_overview.index.to_list() == [0, 1, 2]


def test_using_simple_fake_data():
    document_index: pd.DataFrame = pd.DataFrame(
        columns=['document_id', 'document_name', 'n_tokens'],
        data=[
            [0, 'doc1', 99],
            [1, 'doc2', 99],
            [2, 'doc3', 99],
            [3, 'doc4', 99],
            [4, 'doc5', 99],
        ],
    ).set_index('document_id')

    dictionary: pd.DataFrame = pd.DataFrame(
        columns=['token_id', 'token'],
        data=[
            [0, 'a'],
            [1, 'b'],
            [2, 'c'],
            [3, 'd'],
            [4, 'e'],
        ],
    ).set_index('token_id')

    document_topic_weights: pd.DataFrame = pd.DataFrame(
        columns=['document_id', 'topic_id', 'weight'],
        data=[
            # doc1: a a a a b
            [0, 0, 0.8],
            [0, 1, 0.2],
            # doc2: c c c c d
            [1, 2, 0.8],
            [1, 3, 0.2],
            # doc3: c c c c c c b b b a
            [2, 2, 0.6],
            [2, 1, 0.3],
            [2, 0, 0.1],
            # doc4: d d d d c
            [3, 4, 0.8],
            [3, 3, 0.2],
            # doc5: d
            [4, 4, 1.0],
        ],
    )

    topic_token_weights: pd.DataFrame = pd.DataFrame(
        columns=['topic_id', 'token_id', 'weight'],
        data=[
            # t0: a a a b c
            [0, 0, 0.6],
            [0, 1, 0.2],
            [0, 2, 0.2],
            # t1:  d b c
            [1, 3, 0.8],
            [1, 1, 0.1],
            [1, 2, 0.1],
            # t2:  b b b b b b a a c d
            [2, 1, 0.6],
            [2, 0, 0.2],
            [2, 2, 0.1],
            [2, 3, 0.1],
            # t3:  b b b b b b b b b c
            [3, 1, 0.9],
            [3, 2, 0.1],
            # t3:  a a d e
            [4, 0, 0.4],
            [4, 2, 0.2],
            [4, 4, 0.2],
        ],
    )
    token2id: dict[int, str] = dictionary['token'].to_dict()
    topic_token_overview = ntm.compute_topic_token_overview(topic_token_weights, token2id, 3)
    topic_token_overview['label'] = ['T1', 'T2', 'T3', 'T4', 'T5']
    inferred_topics: ntm.InferredTopicsData = ntm.InferredTopicsData(
        document_index=document_index,
        dictionary=dictionary,
        document_topic_weights=document_topic_weights,
        topic_token_weights=topic_token_weights,
        topic_token_overview=topic_token_overview,
        topic_diagnostics=None,
        token_diagnostics=None,
    )

    assert inferred_topics is not None

    cluster_mapping: dict[str, list[int]] = {
        'C1': [0, 1],
        'C2': [2, 3],
        'C3': [4],
    }
    expected_document_topic_weights = pd.DataFrame(
        columns=['document_id', 'topic_id', 'weight'],
        data=[
            [0, 0, 1.0],
            [1, 2, 1.0],
            [2, 0, 0.4],
            [2, 2, 0.6],
            [3, 2, 0.2],
            [3, 4, 0.8],
            [4, 4, 1.0],
        ],
    )
    inferred_topics.merge(cluster_mapping)

    assert set(inferred_topics.document_topic_weights.topic_id.unique()) == {0, 2, 4}
    assert set(inferred_topics.topic_token_weights.topic_id.unique()) == {0, 2, 4}

    assert inferred_topics.topic_token_overview.label.tolist() == ['C1', 'T2', 'C2', 'T4', 'T5']
    assert inferred_topics.document_topic_weights.equals(expected_document_topic_weights)

    inferred_topics.compress()

    assert set(inferred_topics.document_topic_weights.topic_id.unique()) == {0, 1, 2}
    assert inferred_topics.topic_token_overview.label.tolist() == ['C1', 'C2', 'T5']

    expected_compressed_document_topic_weights = expected_document_topic_weights.copy()
    expected_compressed_document_topic_weights['topic_id'] = (
        expected_compressed_document_topic_weights.topic_id.replace({0: 0, 1: 0, 2: 1, 3: 1, 4: 2})
    )

    assert inferred_topics.document_topic_weights.equals(expected_compressed_document_topic_weights)
