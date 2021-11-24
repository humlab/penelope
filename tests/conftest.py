# type: ignore

import os
import shutil

import pytest
from penelope.notebook.topic_modelling import TopicModelContainer
from penelope.pipeline.spacy import convert
from penelope.topic_modelling import InferredTopicsData
from penelope.vendor.spacy.utility import load_model
from spacy.language import Language
from tests.utils import PERSISTED_INFERRED_MODEL_SOURCE_FOLDER

from .fixtures import MARY_TEST_CORPUS

# pylint: disable=redefined-outer-name


@pytest.fixture(scope="session")
def en_nlp() -> Language:
    return load_model(name_or_nlp="en_core_web_sm", disable="ner")


@pytest.fixture(scope="session")
def df_doc(en_nlp) -> Language:
    attributes = ["text", "lemma_", "pos_", "is_space", "is_punct", "is_digit", "is_alpha", "is_stop"]
    doc = convert.text_to_tagged_frame(
        MARY_TEST_CORPUS[0][1], attributes=attributes, attribute_value_filters=None, nlp=en_nlp
    )
    return doc


# def pytest_configure(config):
#     """
#     Allows plugins and conftest files to perform initial configuration.
#     This hook is called for every plugin and initial conftest
#     file after command line options have been parsed.
#     """


def pytest_sessionstart(session):  # pylint: disable=unused-argument
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    shutil.rmtree('./tests/output', ignore_errors=True)
    os.makedirs('./tests/output', exist_ok=True)


# def pytest_sessionfinish(session, exitstatus):
#     """
#     Called after whole test run finished, right before
#     returning the exit status to the system.
#     """


# def pytest_unconfigure(config):
#     """
#     called before test process is exited.
#     """


@pytest.fixture
def inferred_topics_data() -> InferredTopicsData:
    filename_fields = ["year:_:1", "year_serial_id:_:2"]
    _inferred_topics_data = InferredTopicsData.load(
        folder=PERSISTED_INFERRED_MODEL_SOURCE_FOLDER, filename_fields=filename_fields
    )
    return _inferred_topics_data


@pytest.fixture
def state(inferred_topics_data: InferredTopicsData) -> TopicModelContainer:
    return TopicModelContainer(_trained_model=None, _inferred_topics=inferred_topics_data)
