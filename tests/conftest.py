import os

import penelope.pipeline.spacy.convert as convert
import pytest
import spacy
from spacy.language import Language
from .fixtures import MARY_TEST_CORPUS


@pytest.fixture(scope="session")
def en_nlp() -> Language:
    return spacy.load(os.path.join(os.environ.get("SPACY_DATA", ""), "en_core_web_sm"))


@pytest.fixture(scope="session")
def df_doc(en_nlp) -> Language:
    attributes = ["text", "lemma_", "pos_", "is_space", "is_punct", "is_digit", "is_alpha", "is_stop"]
    doc = convert.text_to_tagged_frame(
        MARY_TEST_CORPUS[0][1], attributes=attributes, attribute_value_filters=None, nlp=en_nlp
    )
    return doc
