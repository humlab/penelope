import os
import shutil
import uuid

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from penelope import corpus as pc
from penelope import topic_modelling as tm
from tests.utils import OUTPUT_FOLDER

jj = os.path.join
isfile = os.path.isfile

# pylint: disable=protected-access,redefined-outer-name


@pytest.fixture
def topics_data() -> tm.InferredTopicsData:
    return tm.InferredTopicsData.load(folder='tests/test_data/tranströmer_inferred_model')


def assert_equal(d1: tm.InferredTopicsData, d2: tm.InferredTopicsData) -> bool:
    assert_frame_equal(d1.document_index, d2.document_index)
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

    pd.testing.assert_frame_equal(topics_data.document_index, loaded_data.document_index, check_dtype=False)

    assert topics_data.topic_token_overview.tokens.tolist() == loaded_data.topic_token_overview.tokens.tolist()
    assert ((loaded_data.topic_token_weights.weight - topics_data.topic_token_weights.weight) < 0.000000005).all()
    assert (loaded_data.topic_token_weights.topic_id == topics_data.topic_token_weights.topic_id).all()
    assert (loaded_data.topic_token_weights.token_id == topics_data.topic_token_weights.token_id).all()
    assert (loaded_data.topic_token_weights.token == topics_data.topic_token_weights.token).all()

    shutil.rmtree(target_folder, ignore_errors=True)


def test_load_token2id(topics_data: tm.InferredTopicsData):
    token2id: pc.Token2Id = tm.InferredTopicsData.load_token2id('tests/test_data/tranströmer_inferred_model')

    assert token2id.id2token == topics_data.id2token
