# type: ignore

import os
import shutil

import pytest
import spacy
from penelope.pipeline.spacy import convert
from penelope.vendor.spacy.utility import download_model
from spacy.language import Language

from .fixtures import MARY_TEST_CORPUS

# pylint: disable=redefined-outer-name


@pytest.fixture(scope="session")
def en_nlp() -> Language:
    model_folder: str = download_model(lang='en', version='2.3.1', folder='./tests/test_data/tmp/')
    return spacy.load(model_folder)


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
    shutil.rmtree('./tests/test_data/output', ignore_errors=True)
    os.makedirs('./tests/test_data/output', exist_ok=True)


# def pytest_sessionfinish(session, exitstatus):
#     """
#     Called after whole test run finished, right before
#     returning the exit status to the system.
#     """


# def pytest_unconfigure(config):
#     """
#     called before test process is exited.
#     """
