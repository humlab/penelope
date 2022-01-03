import functools
import os
import shutil
import uuid

import pandas as pd
import penelope.topic_modelling as topic_modelling
import pytest
from penelope.corpus.dtm.corpus import VectorizedCorpus
from penelope.scripts.tm.train_legacy import main
from penelope.topic_modelling import InferredModel, InferredTopicsData
from penelope.topic_modelling.engines import get_engine_by_model_type
from penelope.topic_modelling.engines.engine_gensim import SUPPORTED_ENGINES
from penelope.topic_modelling.engines.interface import ITopicModelEngine
from tests.fixtures import TranströmerCorpus
from tests.utils import OUTPUT_FOLDER

from ..utils import PERSISTED_INFERRED_MODEL_SOURCE_FOLDER

jj = os.path.join

# pylint: disable=protected-access,redefined-outer-name)


def create_train_corpus() -> topic_modelling.TrainingCorpus:
    corpus: TranströmerCorpus = TranströmerCorpus()
    tc: topic_modelling.TrainingCorpus = topic_modelling.TrainingCorpus(
        corpus=corpus,
        document_index=corpus.document_index,
    )
    return tc


@pytest.fixture
def train_corpus() -> topic_modelling.TrainingCorpus:
    return create_train_corpus()


def _create_inferred_model(method: str, train_corpus: topic_modelling.TrainingCorpus) -> topic_modelling.InferredModel:

    inferred_model: topic_modelling.InferredModel = topic_modelling.train_model(
        train_corpus=train_corpus,
        method=method,
        engine_args={
            'n_topics': 4,
            'passes': 1,
            'random_seed': 42,
            'workers': 1,
            'max_iter': 100,
            'work_folder': f'./tests/output/{uuid.uuid4()}',
        },
    )

    return inferred_model


@functools.lru_cache(maxsize=None)
def create_inferred_model(method) -> topic_modelling.InferredModel:
    train_corpus: topic_modelling.TrainingCorpus = create_train_corpus()
    return _create_inferred_model(method, train_corpus)


def test_tranströmers_corpus():

    corpus = TranströmerCorpus()
    for filename, tokens in corpus:
        assert len(filename) > 0
        assert len(tokens) > 0


def test_create_train_corpus():
    train_corpus: topic_modelling.TrainingCorpus = create_train_corpus()
    assert isinstance(train_corpus.document_index, pd.DataFrame)
    assert len(train_corpus.effective_corpus) == len(train_corpus.document_index)
    assert len(train_corpus.document_index) == 5
    assert len(train_corpus.document_index.columns) == 7
    assert 'n_terms' in train_corpus.document_index.columns


@pytest.mark.parametrize("method", ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_infer_model(method):

    inferred_model = create_inferred_model(method)

    assert inferred_model is not None
    assert inferred_model.method == method

    engine: ITopicModelEngine = get_engine_by_model_type(inferred_model.topic_model)
    assert isinstance(inferred_model.topic_model, engine.supported_models())


def test_load_inferred_model_fixture():

    inferred_model: InferredModel = InferredModel.load(PERSISTED_INFERRED_MODEL_SOURCE_FOLDER)
    assert inferred_model is not None


@pytest.mark.parametrize("method", ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_store_compressed_inferred_model(method):

    inferred_model: InferredModel = create_inferred_model(method)
    target_name = f"{uuid.uuid1()}"
    target_folder = os.path.join(OUTPUT_FOLDER, target_name)

    inferred_model.store(target_folder, store_compressed=False)

    assert VectorizedCorpus.dump_exists(folder=target_folder, tag="train")
    assert os.path.isfile(os.path.join(target_folder, "model_options.json"))

    shutil.rmtree(target_folder)


@pytest.mark.parametrize("method", ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_store_uncompressed_inferred_model(method):

    inferred_model: InferredModel = create_inferred_model(method)
    target_name = f"{uuid.uuid1()}"
    target_folder = os.path.join(OUTPUT_FOLDER, target_name)

    inferred_model.store(target_folder, store_compressed=True)

    # Assert
    assert VectorizedCorpus.dump_exists(folder=target_folder, tag="train")
    assert os.path.isfile(os.path.join(target_folder, "model_options.json"))

    shutil.rmtree(target_folder)


@pytest.mark.parametrize("engine_key", ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_load_inferred_model_when_stored_corpus_is_true_has_same_loaded_trained_corpus(engine_key):

    target_name = f"{uuid.uuid1()}"
    target_folder = os.path.join(OUTPUT_FOLDER, target_name)
    test_inferred_model: InferredModel = create_inferred_model(engine_key)
    test_inferred_model.store(target_folder, store_compressed=True)

    inferred_model: InferredModel = InferredModel.load(target_folder)

    assert inferred_model is not None
    assert inferred_model.method == engine_key
    assert isinstance(inferred_model.topic_model, SUPPORTED_ENGINES[engine_key].engine)

    train_corpus = topic_modelling.TrainingCorpus.load(target_folder)
    assert isinstance(train_corpus.document_index, pd.DataFrame)
    assert len(train_corpus.effective_corpus) == len(train_corpus.document_index)
    assert len(train_corpus.document_index) == 5
    assert len(train_corpus.document_index.columns) == 8
    assert 'n_terms' in train_corpus.document_index.columns
    assert train_corpus.corpus is not None

    shutil.rmtree(target_folder)


@pytest.mark.parametrize("engine_key", ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_load_inferred_model_when_stored_corpus_is_false_has_no_trained_corpus(engine_key):

    target_name: str = f"{uuid.uuid1()}"
    target_folder: str = os.path.join(OUTPUT_FOLDER, target_name)
    test_inferred_model: InferredModel = create_inferred_model(engine_key)
    test_inferred_model.store(target_folder)

    inferred_model = InferredModel.load(target_folder)

    assert inferred_model is not None
    assert inferred_model.method == engine_key
    assert isinstance(inferred_model.topic_model, SUPPORTED_ENGINES[engine_key].engine)

    shutil.rmtree(target_folder)


@pytest.mark.parametrize("method", ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_load_inferred_model_when_lazy_does_not_load_model_or_corpus(method):

    # Arrange
    target_name = f"{uuid.uuid1()}"
    target_folder = jj(OUTPUT_FOLDER, target_name)
    test_inferred_model: InferredModel = create_inferred_model(method)
    test_inferred_model.store(target_folder)

    inferred_model: InferredModel = InferredModel.load(target_folder, lazy=True)

    assert callable(inferred_model._topic_model)

    _ = inferred_model.topic_model

    assert not callable(inferred_model._topic_model)

    _ = inferred_model.topic_model

    shutil.rmtree(target_folder)


@pytest.mark.parametrize("method", ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_infer_topics_data(method, train_corpus):

    minimum_probability: float = 0.001
    n_tokens: int = 5

    inferred_model: InferredModel = create_inferred_model(method)

    inferred_topics_data: InferredTopicsData = topic_modelling.predict_topics(
        topic_model=inferred_model.topic_model,
        corpus=train_corpus.effective_corpus,
        id2token=train_corpus.id2token,
        document_index=train_corpus.document_index,
        minimum_probability=minimum_probability,
        n_tokens=n_tokens,
    )

    assert inferred_topics_data is not None
    assert isinstance(inferred_topics_data.document_index, pd.DataFrame)
    assert isinstance(inferred_topics_data.dictionary, pd.DataFrame)
    assert isinstance(inferred_topics_data.topic_token_weights, pd.DataFrame)
    assert isinstance(inferred_topics_data.topic_token_overview, pd.DataFrame)
    assert isinstance(inferred_topics_data.document_topic_weights, pd.DataFrame)
    assert inferred_topics_data.year_period == (2019, 2020)
    assert inferred_topics_data.topic_ids == [0, 1, 2, 3]
    assert len(inferred_topics_data.document_index) == 5
    assert list(inferred_topics_data.topic_token_weights.topic_id.unique()) == [0, 1, 2, 3]
    assert list(inferred_topics_data.topic_token_overview.index) == [0, 1, 2, 3]
    assert list(inferred_topics_data.document_topic_weights.topic_id.unique()) == [0, 1, 2, 3]


@pytest.mark.parametrize("method", ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_store_inferred_topics_data(method, train_corpus):

    minimum_probability: float = 0.001
    n_tokens: int = 5

    inferred_model: InferredModel = create_inferred_model(method)

    inferred_topics_data: InferredTopicsData = topic_modelling.predict_topics(
        topic_model=inferred_model.topic_model,
        corpus=train_corpus.corpus,
        id2token=train_corpus.id2token,
        document_index=train_corpus.document_index,
        minimum_probability=minimum_probability,
        n_tokens=n_tokens,
    )
    target_folder = jj(OUTPUT_FOLDER, f"{uuid.uuid1()}")

    inferred_topics_data.store(target_folder)

    assert os.path.isfile(jj(target_folder, "dictionary.zip"))
    assert os.path.isfile(jj(target_folder, "document_topic_weights.zip"))
    assert os.path.isfile(jj(target_folder, "documents.zip"))
    assert os.path.isfile(jj(target_folder, "topic_token_overview.zip"))
    assert os.path.isfile(jj(target_folder, "topic_token_weights.zip"))

    shutil.rmtree(target_folder)


@pytest.mark.parametrize("method", ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_load_inferred_topics_data(method, train_corpus):

    minimum_probability: float = 0.001
    n_tokens: int = 5

    inferred_model: InferredModel = create_inferred_model(method)

    test_inferred_topics_data: InferredTopicsData = topic_modelling.predict_topics(
        topic_model=inferred_model.topic_model,
        corpus=train_corpus.effective_corpus,
        id2token=train_corpus.id2token,
        document_index=train_corpus.document_index,
        minimum_probability=minimum_probability,
        n_tokens=n_tokens,
    )
    target_folder = jj(OUTPUT_FOLDER, f"{uuid.uuid1()}")
    test_inferred_topics_data.store(target_folder)

    inferred_topics_data: InferredTopicsData = topic_modelling.InferredTopicsData.load(
        folder=target_folder, filename_fields=None
    )

    assert inferred_topics_data is not None
    assert inferred_topics_data.dictionary.equals(test_inferred_topics_data.dictionary)

    pd.testing.assert_frame_equal(
        inferred_topics_data.document_index, test_inferred_topics_data.document_index, check_dtype=False
    )

    assert (
        test_inferred_topics_data.topic_token_overview.alpha.tolist()
        == inferred_topics_data.topic_token_overview.alpha.tolist()
    )
    assert (
        test_inferred_topics_data.topic_token_overview.tokens.tolist()
        == inferred_topics_data.topic_token_overview.tokens.tolist()
    )

    assert (
        (inferred_topics_data.topic_token_weights.weight - test_inferred_topics_data.topic_token_weights.weight)
        < 0.000000005
    ).all()
    assert (
        inferred_topics_data.topic_token_weights.topic_id == test_inferred_topics_data.topic_token_weights.topic_id
    ).all()
    assert (
        inferred_topics_data.topic_token_weights.token_id == test_inferred_topics_data.topic_token_weights.token_id
    ).all()
    assert (inferred_topics_data.topic_token_weights.token == test_inferred_topics_data.topic_token_weights.token).all()
    shutil.rmtree(target_folder, ignore_errors=True)


@pytest.mark.parametrize("method", ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_run_cli(method):

    kwargs = {
        'target_name': f"{uuid.uuid1()}",
        'corpus_folder': OUTPUT_FOLDER,
        'corpus_source': './tests/test_data/test_corpus.zip',
        'engine': method,
        'engine_args': {
            'n_topics': 5,
            'alpha': 'asymmetric',
        },
        'filename_field': ('year:_:1', 'sequence_id:_:2'),
    }

    main(**kwargs)

    target_folder = jj(kwargs['corpus_folder'], kwargs['target_name'])

    assert os.path.isdir(target_folder)
    assert os.path.isfile(jj(target_folder, 'topic_model.pickle.pbz2'))
    assert os.path.isfile(jj(target_folder, 'model_options.json'))

    inferred_model: InferredModel = InferredModel.load(target_folder)
    inferred_topic_data = InferredTopicsData.load(folder=target_folder, filename_fields=None)

    assert inferred_model is not None
    assert inferred_topic_data is not None

    shutil.rmtree(target_folder)
