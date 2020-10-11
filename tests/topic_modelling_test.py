import os
import shutil
import uuid

import pytest  # pylint: disable=unused-import
import gensim
import pandas as pd

import penelope.topic_modelling as topic_modelling
from penelope.topic_modelling.container import InferredTopicsData, TrainingCorpus
from tests.test_data.tranströmer_corpus import TranströmerCorpus

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

OUTPUT_FOLDER = "./tests/output/"


def compute_inferred_model():

    engine = "gensim_lda"

    corpus = TranströmerCorpus()
    train_corpus = TrainingCorpus(
        terms=corpus.terms,
        documents=corpus.documents,
    )

    inferred_model = topic_modelling.infer_model(
        train_corpus=train_corpus,
        method=engine,
        engine_args=TOPIC_MODELING_OPTS,
    )
    return inferred_model


def test_tranströmers_corpus():

    corpus = TranströmerCorpus()
    for filename, tokens in corpus:
        assert len(filename) > 0
        assert len(tokens) > 0


def test_infer_model():

    inferred_model = compute_inferred_model()

    assert inferred_model is not None
    assert inferred_model.method == "gensim_lda"
    assert isinstance(inferred_model.topic_model, gensim.models.ldamodel.LdaModel)
    assert inferred_model.options["engine_options"] == TOPIC_MODELING_OPTS
    assert isinstance(inferred_model.train_corpus.documents, pd.DataFrame)
    assert len(inferred_model.train_corpus.corpus) == len(inferred_model.train_corpus.documents)
    assert len(inferred_model.train_corpus.documents) == 5
    assert len(inferred_model.train_corpus.documents.columns) == 6
    assert 'n_terms' in inferred_model.train_corpus.documents.columns
    assert inferred_model.train_corpus.corpus is not None


def test_store_inferred_model():

    # Arrange
    name = f"{uuid.uuid1()}"
    target_folder = os.path.join(OUTPUT_FOLDER, name)
    inferred_model = compute_inferred_model()

    # Act
    topic_modelling.store_model(inferred_model, target_folder)

    # Assert
    assert os.path.isfile(os.path.join(target_folder, "inferred_model.pickle"))

    shutil.rmtree(target_folder)


def test_load_inferred_model_when_stored_corpus_is_true_has_same_loaded_trained_corpus():

    # Arrange
    name = f"{uuid.uuid1()}"
    target_folder = os.path.join(OUTPUT_FOLDER, name)
    test_inferred_model = compute_inferred_model()
    topic_modelling.store_model(test_inferred_model, target_folder, store_corpus=True)

    # Act

    inferred_model = topic_modelling.load_model(target_folder)

    # Assert
    assert inferred_model is not None
    assert inferred_model.method == "gensim_lda"
    assert isinstance(inferred_model.topic_model, gensim.models.ldamodel.LdaModel)
    assert inferred_model.options["engine_options"] == TOPIC_MODELING_OPTS
    assert isinstance(inferred_model.train_corpus.documents, pd.DataFrame)
    assert len(inferred_model.train_corpus.corpus) == len(inferred_model.train_corpus.documents)
    assert len(inferred_model.train_corpus.documents) == 5
    assert len(inferred_model.train_corpus.documents.columns) == 6
    assert 'n_terms' in inferred_model.train_corpus.documents.columns
    assert inferred_model.train_corpus.corpus is not None

    shutil.rmtree(target_folder)


def test_load_inferred_model_when_stored_corpus_is_false_has_no_trained_corpus():

    # Arrange
    name = f"{uuid.uuid1()}"
    target_folder = os.path.join(OUTPUT_FOLDER, name)
    test_inferred_model = compute_inferred_model()
    topic_modelling.store_model(test_inferred_model, target_folder, store_corpus=False)

    # Act

    inferred_model = topic_modelling.load_model(target_folder)

    # Assert
    assert inferred_model is not None
    assert inferred_model.method == "gensim_lda"
    assert isinstance(inferred_model.topic_model, gensim.models.ldamodel.LdaModel)
    assert inferred_model.options["engine_options"] == TOPIC_MODELING_OPTS
    assert inferred_model.train_corpus is None

    shutil.rmtree(target_folder)


def test_infer_topics_data():

    # Arrange
    inferred_model = compute_inferred_model()

    # Act

    inferred_topics_data = topic_modelling.compile_inferred_topics_data(
        topic_model=inferred_model.topic_model,
        corpus=inferred_model.train_corpus.corpus,
        id2word=inferred_model.train_corpus.id2word,
        documents=inferred_model.train_corpus.documents,
        n_tokens=5,
    )

    # Assert
    assert inferred_topics_data is not None
    assert isinstance(inferred_topics_data.documents, pd.DataFrame)
    assert isinstance(inferred_topics_data.dictionary, pd.DataFrame)
    assert isinstance(inferred_topics_data.topic_token_weights, pd.DataFrame)
    assert isinstance(inferred_topics_data.topic_token_overview, pd.DataFrame)
    assert isinstance(inferred_topics_data.document_topic_weights, pd.DataFrame)
    assert inferred_topics_data.year_period == (2019, 2020)
    assert inferred_topics_data.topic_ids == [0, 1, 2, 3]
    assert len(inferred_topics_data.documents) == 5
    assert list(inferred_topics_data.topic_token_weights.topic_id.unique()) == [0, 1, 2, 3]
    assert list(inferred_topics_data.topic_token_overview.index) == [0, 1, 2, 3]
    assert list(inferred_topics_data.document_topic_weights.topic_id.unique()) == [0, 1, 2, 3]


def test_store_inferred_topics_data():

    # Arrange
    inferred_model = compute_inferred_model()

    inferred_topics_data: InferredTopicsData = topic_modelling.compile_inferred_topics_data(
        topic_model=inferred_model.topic_model,
        corpus=inferred_model.train_corpus.corpus,
        id2word=inferred_model.train_corpus.id2word,
        documents=inferred_model.train_corpus.documents,
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


def test_load_inferred_topics_data():

    # Arrange
    inferred_model = compute_inferred_model()

    test_inferred_topics_data: InferredTopicsData = topic_modelling.compile_inferred_topics_data(
        topic_model=inferred_model.topic_model,
        corpus=inferred_model.train_corpus.corpus,
        id2word=inferred_model.train_corpus.id2word,
        documents=inferred_model.train_corpus.documents,
        n_tokens=5,
    )
    target_folder = jj(OUTPUT_FOLDER, f"{uuid.uuid1()}")
    test_inferred_topics_data.store(target_folder)

    # Act
    inferred_topics_data: InferredTopicsData = topic_modelling.InferredTopicsData.load(target_folder)

    # Assert
    assert inferred_topics_data is not None
    assert inferred_topics_data.dictionary.equals(test_inferred_topics_data.dictionary)
    assert inferred_topics_data.documents.equals(test_inferred_topics_data.documents)
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
