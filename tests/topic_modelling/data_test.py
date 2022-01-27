import os
import shutil
import uuid

import pandas as pd
import pytest

from penelope import topic_modelling as tm
from penelope.topic_modelling.topics_data.topics_data import InferredTopicsData
from tests.utils import OUTPUT_FOLDER

jj = os.path.join
isfile = os.path.isfile

# pylint: disable=protected-access,redefined-outer-name


@pytest.fixture
def topics_data() -> InferredTopicsData:
    return InferredTopicsData.load(folder='tests/test_data/tranströmer_inferred_model')


def test_load_topics_data(topics_data: InferredTopicsData):

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
def test_store_inferred_topics_data_as_zipped_files(topics_data: InferredTopicsData, format: str):

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

    loaded_data: InferredTopicsData = InferredTopicsData.load(folder=target_folder)

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
