from unittest.mock import MagicMock, Mock, patch

import pytest

from penelope.pipeline import ContentType, CorpusConfig, CorpusPipeline, DocumentPayload, ITask, PipelinePayload
from penelope.pipeline import tagged_frame as tagged_frame_tasks
from penelope.pipeline.spacy import tasks
from penelope.pipeline.spacy.tagger import SpacyTagger
from penelope.utility.pos_tags import PoS_Tag_Schemes
from penelope.vendor import spacy_api

SAMPLE_TEXT = "Looking back. Looking back to see if someone is looking back at me."

# pylint: disable=redefined-outer-name


@pytest.fixture
def looking_back(en_nlp) -> spacy_api.Doc:
    pytest.importorskip("spacy")
    return en_nlp(SAMPLE_TEXT)


POS_ATTRIBUTES = ['text', 'pos_', 'lemma_']
MEMORY_STORE = {'lang': 'en', 'lemma_column': 'lemma_', 'pos_column': 'pos_', 'text_column': 'text', 'tagger': {}}


@pytest.fixture
def test_payload():
    mock = Mock(
        spec=PipelinePayload, memory_store=MEMORY_STORE, pos_schema=PoS_Tag_Schemes.Universal
    )  # document_index=MagicMock(pd.DataFrame),
    return mock


def patch_spacy_load(*x, **y):  # pylint: disable=unused-argument
    return MagicMock(spec=spacy_api.Language, return_value=MagicMock(spec=spacy_api.Doc))


def patch_spacy_pipeline(payload: PipelinePayload):
    config: MagicMock = MagicMock(spec=CorpusConfig, pipeline_payload=payload)
    pipeline: CorpusPipeline = CorpusPipeline(config=config, tasks=[], payload=payload).setup()
    return pipeline


@pytest.mark.skipif(not spacy_api.SPACY_INSTALLED, reason="Spacy not installed")
@patch('spacy.load', patch_spacy_load)
def test_to_spacy_doc(test_payload, tagger: SpacyTagger):
    pytest.importorskip("spacy")
    task = tasks.ToSpacyDoc(tagger=tagger)
    _ = patch_spacy_pipeline(test_payload).setup()
    payload = DocumentPayload(content_type=ContentType.TEXT, filename='hello.txt', content="Hello world!")
    payload_next = task.process_payload(payload)
    assert payload_next.content_type == ContentType.SPACYDOC


@pytest.mark.skipif(not spacy_api.SPACY_INSTALLED, reason="Spacy not installed")
@patch('spacy.load', patch_spacy_load)
def test_spacy_doc_to_tagged_frame(looking_back, test_payload, tagger):
    pytest.importorskip("spacy")
    payload = DocumentPayload(content_type=ContentType.SPACYDOC, filename='hello.txt', content=looking_back)
    prior = Mock(spec=ITask, outstream=lambda: [payload])
    task = tagged_frame_tasks.ToTaggedFrame(prior=prior, tagger=tagger)
    task.register_pos_counts = lambda p: p
    _ = patch_spacy_pipeline(test_payload).add([task]).setup()
    payload_next = task.process_payload(payload)
    assert payload_next.content_type == ContentType.TAGGED_FRAME


@pytest.mark.skipif(not spacy_api.SPACY_INSTALLED, reason="Spacy not installed")
@patch('spacy.load', patch_spacy_load)
@patch('penelope.pipeline.spacy.convert.filter_tokens_by_attribute_values', lambda *_, **__: ['a'])
def test_to_spacy_doc_to_tagged_frame(test_payload, tagger):
    payload = DocumentPayload(content_type=ContentType.TEXT, filename='hello.txt', content=SAMPLE_TEXT)
    config: CorpusConfig = CorpusConfig.load('./tests/test_data/SSI/SSI.yml')
    pipeline: CorpusPipeline = CorpusPipeline(config=config, tasks=[], payload=payload).setup()
    prior = MagicMock(spec=ITask, outstream=lambda: [payload])
    task = tagged_frame_tasks.ToTaggedFrame(pipeline=pipeline, prior=prior, tagger=tagger)
    task.register_pos_counts = lambda p: p
    _ = patch_spacy_pipeline(test_payload).add([task]).setup()
    payload_next = task.process_payload(payload)
    assert payload_next.content_type == ContentType.TAGGED_FRAME
