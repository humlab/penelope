from typing import List

import pandas as pd
import pytest
import spacy
from penelope.pipeline.spacy.convert import (
    filter_tokens_by_attribute_values,
    spacy_doc_to_tagged_frame,
    text_to_tagged_frame,
    texts_to_tagged_frames,
)
from penelope.vendor.spacy import prepend_spacy_path
from spacy.tokens import Doc, Token

SAMPLE_TEXT = "Looking back. Looking back to see if someone is looking back at me."

# pylint: disable=redefined-outer-name


@pytest.fixture
def nlp() -> Doc:
    return spacy.load(prepend_spacy_path("en_core_web_sm"))


@pytest.fixture
def looking_back(nlp) -> Doc:
    return nlp(SAMPLE_TEXT)


def test_filter_tokens_by_attribute_values(looking_back: Doc):  # pylint: disable=unused-argument

    words = SAMPLE_TEXT.replace('.', ' .').split()

    tokens: List[Token] = list(filter_tokens_by_attribute_values(looking_back, None))

    assert [t.text for t in tokens] == words

    tokens: List[Token] = list(filter_tokens_by_attribute_values(looking_back, {'is_punct': True}))
    assert [t.text for t in tokens] == ['.', '.']

    tokens: List[Token] = list(filter_tokens_by_attribute_values(looking_back, {'is_punct': False}))
    assert [t.text for t in tokens] == [w for w in words if w != '.']

    tokens: List[Token] = list(filter_tokens_by_attribute_values(looking_back, {'is_stop': False, 'is_punct': False}))
    assert [t.text for t in tokens] == ['Looking', 'Looking', 'looking']


def assert_test_xyz_to_tagged_frame(tagged_frame: pd.DataFrame):
    words = ['Looking', 'back', 'Looking', 'back', 'to', 'see', 'if', 'someone', 'is', 'looking', 'back', 'at', 'me']
    pos = ['VERB', 'ADV', 'VERB', 'ADV', 'PART', 'VERB', 'SCONJ', 'PRON', 'AUX', 'VERB', 'ADV', 'ADP', 'PRON']
    lemma = ['look', 'back', 'look', 'back', 'to', 'see', 'if', 'someone', 'be', 'look', 'back', 'at', '-PRON-']
    assert tagged_frame.text.tolist() == words
    assert tagged_frame.pos_.tolist() == pos
    assert tagged_frame.lemma_.tolist() == lemma


def test_spacy_doc_to_tagged_frame(looking_back: Doc):
    tagged_frame = spacy_doc_to_tagged_frame(
        spacy_doc=looking_back,
        attributes=['text', 'pos_', 'lemma_'],
        attribute_value_filters={'is_stop': None, 'is_punct': False},
    )

    assert_test_xyz_to_tagged_frame(tagged_frame)


def test_text_to_tagged_frames(nlp):
    tagged_frame = text_to_tagged_frame(
        document=SAMPLE_TEXT,
        attributes=['text', 'pos_', 'lemma_'],
        attribute_value_filters={'is_stop': None, 'is_punct': False},
        nlp=nlp,
    )
    assert_test_xyz_to_tagged_frame(tagged_frame)


def test_texts_to_tagged_frames(nlp):
    stream = [SAMPLE_TEXT, SAMPLE_TEXT]
    tagged_frames = texts_to_tagged_frames(
        stream=stream,
        attributes=['text', 'pos_', 'lemma_'],
        attribute_value_filters={'is_stop': None, 'is_punct': False},
        language=nlp,
    )
    for tagged_frame in tagged_frames:
        assert_test_xyz_to_tagged_frame(tagged_frame)
