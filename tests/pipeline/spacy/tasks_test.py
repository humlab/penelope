from unittest.mock import MagicMock, Mock, patch

import pytest
import spacy
from penelope.pipeline import ContentType, DocumentPayload, PipelinePayload
from penelope.pipeline.spacy.pipelines import SpacyPipeline
from penelope.pipeline.spacy.tasks import SetSpacyModel, SpacyDocToTaggedFrame, ToSpacyDoc, ToSpacyDocToTaggedFrame
from spacy.tokens import Doc

SAMPLE_TEXT = "Looking back. Looking back to see if someone is looking back at me."

# pylint: disable=redefined-outer-name


@pytest.fixture
def nlp() -> Doc:
    return spacy.load("en_core_web_sm")


@pytest.fixture
def looking_back(nlp) -> Doc:
    return nlp(SAMPLE_TEXT)


POS_ATTRIBUTES = ['text', 'pos_', 'lemma_']
MEMORY_STORE = {
    'lang': 'en',
    'lemma_column': 'lemma_',
    'pos_column': 'pos_',
    'spacy_model': 'en_core_web_sm',
    'tagger': 'spaCy',
    'text_column': 'text',
}


@pytest.fixture
def test_payload():
    mock = Mock(spec=PipelinePayload, memory_store=MEMORY_STORE)  # document_index=MagicMock(pd.DataFrame),
    return mock


def patch_spacy_load(*x, **y):  # pylint: disable=unused-argument
    return MagicMock(spec=spacy.language.Language, return_value=Mock(spec=spacy.tokens.Doc))


def patch_spacy_pipeline(payload: PipelinePayload):
    pipeline = SpacyPipeline(payload=payload, tasks=[]).setup()
    return pipeline


@patch('spacy.load', patch_spacy_load)
def test_set_spacy_model(test_payload):
    task = SetSpacyModel(lang_or_nlp="en")
    pipeline = patch_spacy_pipeline(test_payload)
    pipeline.add(task).setup()
    assert pipeline.get("spacy_nlp") is not None


@patch('spacy.load', patch_spacy_load)
def test_to_spacy_doc(test_payload):
    task = ToSpacyDoc()
    _ = patch_spacy_pipeline(test_payload).add(SetSpacyModel(lang_or_nlp="en")).add(task).setup()
    payload = DocumentPayload(content_type=ContentType.TEXT, filename='hello.txt', content="Hello world!")
    payload_next = task.process_payload(payload)
    assert payload_next.content_type == ContentType.SPACYDOC


@patch('spacy.load', patch_spacy_load)
@patch('penelope.pipeline.convert.tagged_frame_to_token_counts', return_value={})
def test_spacy_doc_to_tagged_frame(looking_back, test_payload):
    payload = DocumentPayload(content_type=ContentType.SPACYDOC, filename='hello.txt', content=looking_back)
    task = SpacyDocToTaggedFrame(instream=[payload], attributes=POS_ATTRIBUTES)
    _ = patch_spacy_pipeline(test_payload).add([SetSpacyModel(lang_or_nlp="en"), task]).setup()
    payload_next = task.process_payload(payload)
    assert payload_next.content_type == ContentType.TAGGEDFRAME


@patch('spacy.load', patch_spacy_load)
@patch('penelope.pipeline.convert.tagged_frame_to_token_counts', return_value={})
def test_to_spacy_doc_to_tagged_frame(test_payload):
    payload = DocumentPayload(content_type=ContentType.TEXT, filename='hello.txt', content=SAMPLE_TEXT)
    task = ToSpacyDocToTaggedFrame(instream=[payload], attributes=POS_ATTRIBUTES)
    _ = patch_spacy_pipeline(test_payload).add([SetSpacyModel(lang_or_nlp="en"), task]).setup()
    payload_next = task.process_payload(payload)
    assert payload_next.content_type == ContentType.TAGGEDFRAME


# @patch('spacy.load', patch_spacy_load)
# @patch('penelope.pipeline.convert.tagged_frame_to_token_counts', return_value={})
# def test_shortcuts(nlp, test_payload):
#     payload = DocumentPayload(content_type=ContentType.TEXT, filename='hello.txt', content=SAMPLE_TEXT)
#     pipeline = patch_spacy_pipeline(test_payload).set_spacy_model(language=nlp).text_to_spacy_to_tagged_frame().setup()
#     pipeline.tasks[1].instream = [payload]
#     payload_next = pipeline.tasks[-1].process_payload(payload)
#     assert payload_next.content_type == ContentType.TAGGEDFRAME
