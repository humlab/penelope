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
