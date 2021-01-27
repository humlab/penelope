import os
import shutil
import uuid

import gensim
import pandas as pd
import penelope.topic_modelling as topic_modelling
import pytest
from penelope.scripts.compute_topic_model import run_model
from penelope.topic_modelling.container import InferredModel, InferredTopicsData, TrainingCorpus
from tests.test_data.tranströmer_corpus import TranströmerCorpus
from tests.utils import OUTPUT_FOLDER

jj = os.path.join

TOPIC_MODELING_OPTS = {
    'n_topics': 4,
    'passes': 1,
    'random_seed': 42,
    'alpha': 'auto',
    'workers': 1,
    'max_iter': 100,
    'prefix': '',
}
# pylint: disable=protected-access


def compute_inferred_model(method="gensim_lda-multicore") -> InferredModel:

    corpus = TranströmerCorpus()
    train_corpus = TrainingCorpus(
        terms=corpus.terms,
        document_index=corpus.document_index,
    )

    inferred_model: InferredModel = topic_modelling.infer_model(
        train_corpus=train_corpus,
        method=method,
        engine_args=TOPIC_MODELING_OPTS,
    )
    return inferred_model


def test_tranströmers_corpus():

    corpus = TranströmerCorpus()
    for filename, tokens in corpus:
        assert len(filename) > 0
        assert len(tokens) > 0


@pytest.mark.parametrize("method", ["gensim_lda-multicore"])
def test_infer_model(method):

    inferred_model = compute_inferred_model(method)
    assert inferred_model is not None
    assert inferred_model.method == method
    assert isinstance(inferred_model.topic_model, gensim.models.ldamodel.LdaModel)
    assert inferred_model.options["engine_options"] == TOPIC_MODELING_OPTS
    assert isinstance(inferred_model.train_corpus.document_index, pd.DataFrame)
    assert len(inferred_model.train_corpus.corpus) == len(inferred_model.train_corpus.document_index)
    assert len(inferred_model.train_corpus.document_index) == 5
    assert len(inferred_model.train_corpus.document_index.columns) == 7
    assert 'n_terms' in inferred_model.train_corpus.document_index.columns
    assert inferred_model.train_corpus.corpus is not None


def test_store_compressed_inferred_model():

    # Arrange
    name = f"{uuid.uuid1()}"
    target_folder = os.path.join(OUTPUT_FOLDER, name)
    inferred_model = compute_inferred_model()

    # Act
    topic_modelling.store_model(inferred_model, target_folder, store_corpus=True, store_compressed=True)

    # Assert
    assert os.path.isfile(os.path.join(target_folder, "topic_model.pickle.pbz2"))
    assert os.path.isfile(os.path.join(target_folder, "training_corpus.pickle.pbz2"))
    assert os.path.isfile(os.path.join(target_folder, "model_options.json"))

    shutil.rmtree(target_folder)


def test_store_uncomressed_inferred_model():

    # Arrange
    name = f"{uuid.uuid1()}"
    target_folder = os.path.join(OUTPUT_FOLDER, name)
    inferred_model = compute_inferred_model()

    # Act
    topic_modelling.store_model(inferred_model, target_folder, store_corpus=True, store_compressed=False)

    # Assert
    assert os.path.isfile(os.path.join(target_folder, "topic_model.pickle"))
    assert os.path.isfile(os.path.join(target_folder, "training_corpus.pickle"))
    assert os.path.isfile(os.path.join(target_folder, "model_options.json"))

    shutil.rmtree(target_folder)


@pytest.mark.parametrize("method", ["gensim_lda-multicore"])
def test_load_inferred_model_when_stored_corpus_is_true_has_same_loaded_trained_corpus(method):

    # Arrange
    name = f"{uuid.uuid1()}"
    target_folder = os.path.join(OUTPUT_FOLDER, name)
    test_inferred_model = compute_inferred_model(method)
    topic_modelling.store_model(test_inferred_model, target_folder, store_corpus=True, store_compressed=True)

    # Act

    inferred_model = topic_modelling.load_model(target_folder)

    # Assert
    assert inferred_model is not None
    assert inferred_model.method == method
    assert isinstance(inferred_model.topic_model, gensim.models.ldamodel.LdaModel)
    assert inferred_model.options['engine_options'] == TOPIC_MODELING_OPTS
    assert isinstance(inferred_model.train_corpus.document_index, pd.DataFrame)
    assert len(inferred_model.train_corpus.corpus) == len(inferred_model.train_corpus.document_index)
    assert len(inferred_model.train_corpus.document_index) == 5
    assert len(inferred_model.train_corpus.document_index.columns) == 7
    assert 'n_terms' in inferred_model.train_corpus.document_index.columns
    assert inferred_model.train_corpus.corpus is not None

    shutil.rmtree(target_folder)


@pytest.mark.parametrize("method", ["gensim_lda-multicore"])
def test_load_inferred_model_when_stored_corpus_is_false_has_no_trained_corpus(method):

    # Arrange
    name = f"{uuid.uuid1()}"
    target_folder = os.path.join(OUTPUT_FOLDER, name)
    test_inferred_model = compute_inferred_model(method)
    topic_modelling.store_model(test_inferred_model, target_folder, store_corpus=False)

    # Act

    inferred_model = topic_modelling.load_model(target_folder)

    # Assert
    assert inferred_model is not None
    assert inferred_model.method == method
    assert isinstance(inferred_model.topic_model, gensim.models.ldamodel.LdaModel)
    assert inferred_model.options['engine_options'] == TOPIC_MODELING_OPTS
    assert inferred_model.train_corpus is None

    shutil.rmtree(target_folder)


@pytest.mark.parametrize("method", ["gensim_lda-multicore"])
def test_load_inferred_model_when_lazy_does_not_load_model_or_corpus(method):

    # Arrange
    name = f"{uuid.uuid1()}"
    target_folder = jj(OUTPUT_FOLDER, name)
    test_inferred_model = compute_inferred_model(method)
    topic_modelling.store_model(test_inferred_model, target_folder, store_corpus=False)

    # Act

    inferred_model = topic_modelling.load_model(target_folder, lazy=True)

    # Assert
    assert callable(inferred_model._topic_model)

    _ = inferred_model.topic_model

    assert not callable(inferred_model._topic_model)

    assert inferred_model._train_corpus is None or callable(inferred_model._train_corpus)

    _ = inferred_model.topic_model

    assert inferred_model._train_corpus is None or not callable(inferred_model._topic_model)

    shutil.rmtree(target_folder)


@pytest.mark.parametrize("method", ["gensim_lda-multicore"])
def test_infer_topics_data(method):

    # Arrange
    inferred_model = compute_inferred_model(method)

    # Act

    inferred_topics_data: InferredTopicsData = topic_modelling.compile_inferred_topics_data(
        topic_model=inferred_model.topic_model,
        corpus=inferred_model.train_corpus.corpus,
        id2word=inferred_model.train_corpus.id2word,
        document_index=inferred_model.train_corpus.document_index,
        n_tokens=5,
    )

    # Assert
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


@pytest.mark.parametrize("method", ["gensim_lda-multicore"])
def test_store_inferred_topics_data(method):

    # Arrange
    inferred_model = compute_inferred_model(method)

    inferred_topics_data: InferredTopicsData = topic_modelling.compile_inferred_topics_data(
        topic_model=inferred_model.topic_model,
        corpus=inferred_model.train_corpus.corpus,
        id2word=inferred_model.train_corpus.id2word,
        document_index=inferred_model.train_corpus.document_index,
        n_tokens=5,
    )
    target_folder = jj(OUTPUT_FOLDER, f"{uuid.uuid1()}")

    # Act
    inferred_topics_data.store(target_folder)

    # Assert
    assert os.path.isfile(jj(target_folder, "dictionary.zip"))
    assert os.path.isfile(jj(target_folder, "document_topic_weights.zip"))
    assert os.path.isfile(jj(target_folder, "documents.zip"))
    assert os.path.isfile(jj(target_folder, "topic_token_overview.zip"))
    assert os.path.isfile(jj(target_folder, "topic_token_weights.zip"))

    shutil.rmtree(target_folder)


@pytest.mark.parametrize("method", ["gensim_lda-multicore"])
def test_load_inferred_topics_data(method):

    # Arrange
    inferred_model = compute_inferred_model(method)

    test_inferred_topics_data: InferredTopicsData = topic_modelling.compile_inferred_topics_data(
        topic_model=inferred_model.topic_model,
        corpus=inferred_model.train_corpus.corpus,
        id2word=inferred_model.train_corpus.id2word,
        document_index=inferred_model.train_corpus.document_index,
        n_tokens=5,
    )
    target_folder = jj(OUTPUT_FOLDER, f"{uuid.uuid1()}")
    test_inferred_topics_data.store(target_folder)

    # Act
    inferred_topics_data: InferredTopicsData = topic_modelling.InferredTopicsData.load(folder=target_folder, filename_fields=None)

    # Assert
    assert inferred_topics_data is not None
    assert inferred_topics_data.dictionary.equals(test_inferred_topics_data.dictionary)
    assert inferred_topics_data.document_index.equals(test_inferred_topics_data.document_index)
    assert inferred_topics_data.topic_token_overview.round(5).equals(
        test_inferred_topics_data.topic_token_overview.round(5)
    )
    # assert inferred_topics_data.document_topic_weights.round(5).equals(test_inferred_topics_data.document_topic_weights.round(5))
    assert (
        inferred_topics_data.topic_token_weights.round(5)
        .eq(test_inferred_topics_data.topic_token_weights.round(5))
        .all()
        .all()
    )
    # assert inferred_topics_data.topic_token_weights.round(5).equals(test_inferred_topics_data.topic_token_weights.round(5))
    assert (
        inferred_topics_data.topic_token_weights.round(5)
        .eq(test_inferred_topics_data.topic_token_weights.round(5))
        .all()
        .all()
    )

    shutil.rmtree(target_folder)


def test_run_cli():

    kwargs = {
        'name': f"{uuid.uuid1()}",
        'n_topics': 5,
        'corpus_folder': OUTPUT_FOLDER,
        'corpus_filename': './tests/test_data/test_corpus.zip',
        'engine': 'gensim_lda-multicore',
        # 'passes': None,
        # 'random_seed': None,
        'alpha': 'asymmetric',
        # 'workers': None,
        # 'max_iter': None,
        # 'prefix': None,
        'filename_field': ('year:_:1', 'sequence_id:_:2'),
    }

    run_model(**kwargs)

    target_folder = jj(kwargs['corpus_folder'], kwargs['name'])

    assert os.path.isdir(target_folder)
    assert os.path.isfile(jj(target_folder, 'topic_model.pickle.pbz2'))
    assert os.path.isfile(jj(target_folder, 'model_options.json'))

    shutil.rmtree(target_folder)


def test_run_model_by_cli_stores_a_model_that_can_be_loaded():

    name = "test_corpus.xyz"
    target_folder = jj(OUTPUT_FOLDER, name)
    options = dict(
        name=name,
        n_topics=5,
        corpus_folder=OUTPUT_FOLDER,
        corpus_filename='./tests/test_data/test_corpus.zip',
        engine="gensim_lda-multicore",
        workers=2,
        max_iter=2000,
        store_corpus=True,
        filename_field="year:_:1",
    )

    run_model(**options)

    inferred_model = topic_modelling.load_model(target_folder)
    inferred_topic_data = InferredTopicsData.load(folder=target_folder, filename_fields=None)

    assert inferred_model is not None
    assert inferred_topic_data is not None

    shutil.rmtree(target_folder)
