import os
from typing import Iterable, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
import spacy.language
import spacy.tokens
from penelope.corpus.readers import TextReaderOpts, TextTransformOpts
from penelope.corpus.readers.interfaces import ExtractTokensOpts2
from penelope.vendor.spacy import ContentType, CorpusPipeline, DocumentPayload, PipelinePayload, SpacyPipeline, tasks

TEST_CORPUS = [
    ('mary_1859_01.txt', 'Mary had a little lamb. Its fleece was white as snow.'),
    ('mary_1859_02.txt', 'Everywhere that Mary went the lamb was sure to go.'),
    ('mary_1859_03.txt', 'It followed her to school one day, which was against the rule.'),
    ('mary_1859_04.txt', 'It made the children laugh and play to see a lamb at school.'),
    ('mary_1860_05.txt', 'And so the teacher turned it out, but still it lingered near.'),
    ('mary_1860_06.txt', 'It waited patiently about until Mary did appear.'),
    ('mary_1861_07.txt', 'Why does the lamb love Mary so? The eager children cry.'),
    ('mary_1861_08.txt', 'Mary loves the lamb, you know, the teacher did reply.'),
]

TEST_OUTPUT_FOLDER = './tests/output'


@pytest.fixture(scope="module")
def mary_had_a_little_lamb_corpus() -> Iterable[Tuple[str, str]]:
    return TEST_CORPUS


@pytest.fixture(scope="module")
def reader_opts():
    return TextReaderOpts(
        filename_fields=["document_id:_:2", "year:_:1"],
        filename_fields_key="document_id",
        filename_filter=None,
        filename_pattern="*.txt",
        as_binary=False,
    )


def fake_data_frame_stream(n: int = 1):
    df_dummy = pd.DataFrame(data={'text': ['bil'], 'pos_': ['NOUN'], 'lemma_': ['bil']})
    for i in range(1, n + 1):
        yield DocumentPayload(
            filename=f'dummy_{i}.csv',
            content_type=ContentType.DATAFRAME,
            content=df_dummy,
        )


def fake_spacy_doc_stream(n: int = 1):
    dummy = MagicMock(spec=spacy.tokens.Doc)
    for i in range(1, n + 1):
        yield DocumentPayload(
            filename=f'dummy_{i}.txt',
            content_type=ContentType.SPACYDOC,
            content=dummy,
        )


def fake_text_stream(n: int = 1):
    for i in (0, n):
        yield DocumentPayload(filename=TEST_CORPUS[i][0], content_type=ContentType.TEXT, content=TEST_CORPUS[i][1])


def fake_token_stream(n: int = 1):
    for i in (0, n):
        yield DocumentPayload(
            filename=TEST_CORPUS[i][0], content_type=ContentType.TOKENS, content=TEST_CORPUS[i][1].split()
        )


def patch_spacy_load(*x, **y):  # pylint: disable=unused-argument
    mock_doc = Mock(spec=spacy.tokens.Doc)
    m = MagicMock(spec=spacy.language.Language)
    m.return_value = mock_doc
    return m


def patch_spacy_doc(*x, **y):  # pylint: disable=unused-argument
    m = MagicMock(spec=spacy.tokens.Doc)
    return m


def patch_spacy_pipeline(task):
    pipeline = SpacyPipeline(
        payload=PipelinePayload(memory_store=dict(spacy_nlp=patch_spacy_load())),
        tasks=[task],
    ).setup()
    return pipeline


@patch('spacy.load', patch_spacy_load)
def test_set_spacy_model_setup_succeeds():
    pipeline = SpacyPipeline(payload=PipelinePayload())
    task = tasks.SetSpacyModel(pipeline=pipeline, language="en_core_web_sm").setup()
    assert pipeline.get("spacy_nlp", None) is not None


def test_load_text_when_source_is_list_of_filename_text_tuples_succeeds(reader_opts):
    transform_opts = TextTransformOpts()
    pipeline: CorpusPipeline = Mock(spec=CorpusPipeline, payload=Mock(spec=PipelinePayload, document_index=None))
    task = tasks.LoadText(
        pipeline=pipeline, source=TEST_CORPUS, reader_opts=reader_opts, transform_opts=transform_opts
    ).setup()
    current_payload = next(fake_text_stream())
    next_payload = task.process(current_payload)
    assert next_payload.content_type == ContentType.TEXT
    assert next_payload.content == 'Mary had a little lamb. Its fleece was white as snow.'
    assert next_payload.filename == 'mary_1859_01.txt'


def test_tqdm_task():
    pipeline = Mock(spec=CorpusPipeline, payload=Mock(spec=PipelinePayload, document_index=MagicMock(pd.DataFrame)))
    task = tasks.Tqdm(pipeline=pipeline).setup()
    current_payload = next(fake_text_stream(1))
    next_payload = task.process(current_payload)
    assert current_payload == next_payload


def test_passthrough_process_succeeds():
    task = tasks.Passthrough(pipeline=CorpusPipeline(payload=PipelinePayload())).setup()
    current_payload = DocumentPayload()
    next_payload = task.process(current_payload)
    assert current_payload == next_payload


def test_project_process_with_text_payload_succeeds():
    def project(p: DocumentPayload):
        p.content = "HELLO"
        return p

    pipeline = Mock(spec=CorpusPipeline)
    task = tasks.Project(pipeline=pipeline, project=project).setup()
    current_payload = next(fake_text_stream(1))
    next_payload = task.process(current_payload)
    assert next_payload.content == "HELLO"


def test_to_content_process_with_text_payload_succeeds():

    task = tasks.ToContent(pipeline=Mock(spec=CorpusPipeline)).setup()
    current_payload = next(fake_text_stream(1))
    next_payload = task.process(current_payload)
    assert next_payload == TEST_CORPUS[0][1]


def test_text_to_spacy_process_with_text_payload_succeeds():
    task = tasks.TextToSpacy(pipeline=Mock(spec=CorpusPipeline)).setup()
    current_payload = next(fake_text_stream())
    next_payload = task.process(current_payload)
    assert next_payload.content_type == ContentType.SPACYDOC


def test_text_to_spacy_process_with_non_text_payload_fails():
    task = tasks.TextToSpacy(pipeline=Mock(spec=CorpusPipeline)).setup()
    current_payload = next(fake_data_frame_stream(1))
    with pytest.raises(Exception) as _:
        _ = task.setup().process(current_payload)


def patch_any_to_annotated_dataframe(*_) -> pd.DataFrame:
    return pd.DataFrame(data={'text': ['bil'], 'pos_': ['NOUN'], 'lemma_': ['bil']})


def patch_spacy_doc_to_annotated_dataframe(*_, **__) -> pd.DataFrame:
    return pd.DataFrame(data={'text': ['bil'], 'pos_': ['NOUN'], 'lemma_': ['bil']})


@patch('penelope.vendor.spacy.convert.spacy_doc_to_annotated_dataframe', patch_spacy_doc_to_annotated_dataframe)
def test_text_to_dataframe_process_with_text_payload_succeeds():
    task = tasks.TextToSpacyToDataFrame(pipeline=Mock(spec=CorpusPipeline)).setup()
    current_payload = next(fake_text_stream())
    next_payload = task.process(current_payload)
    assert next_payload.content_type == ContentType.DATAFRAME


@patch('penelope.vendor.spacy.convert.spacy_doc_to_annotated_dataframe', patch_any_to_annotated_dataframe)
def test_spacy_to_dataframe_process_with_doc_payload_succeeds():
    task = tasks.SpacyToDataFrame(pipeline=Mock(spec=CorpusPipeline)).setup()
    current_payload = next(fake_spacy_doc_stream())
    next_payload = task.process(current_payload)
    assert next_payload.content_type == ContentType.DATAFRAME
    assert next_payload.content.columns.tolist() == ['text', 'pos_', 'lemma_']


def dataframe_to_tokens_patch(*_) -> Iterable[str]:
    return ["a", "b", "c"]


@patch('penelope.vendor.spacy.convert.dataframe_to_tokens', dataframe_to_tokens_patch)
def test_data_frame_to_tokens_succeeds():
    task = tasks.DataFrameToTokens(pipeline=Mock(spec=CorpusPipeline), extract_word_opts=ExtractTokensOpts2()).setup()
    current_payload = next(fake_data_frame_stream(1))
    next_payload = task.process(current_payload)
    assert next_payload.content_type == ContentType.TOKENS


def patch_store_data_frame_stream(
    *, target_filename, document_index, payload_stream  # pylint: disable=unused-argument
):
    for p in payload_stream:
        yield p


@patch('penelope.vendor.spacy.convert.store_data_frame_stream', patch_store_data_frame_stream)
def test_save_data_frame_succeeds():
    task = tasks.SaveDataFrame(pipeline=Mock(spec=CorpusPipeline), filename="dummy.zip")
    task.instream = fake_data_frame_stream(1)
    for payload in task.outstream():
        assert payload.content_type == ContentType.DATAFRAME


def patch_load_data_frame_stream(*_, **__) -> Tuple[Iterable[DocumentPayload], Optional[pd.DataFrame]]:
    return fake_data_frame_stream(1), None


@patch('penelope.vendor.spacy.convert.load_data_frame_stream', patch_load_data_frame_stream)
def test_load_data_frame_succeeds():
    task = tasks.LoadDataFrame(pipeline=Mock(spec=CorpusPipeline), filename="dummy.zip").setup()
    task.instream = fake_data_frame_stream(1)
    for payload in task.outstream():
        assert payload.content_type == ContentType.DATAFRAME


@patch('penelope.vendor.spacy.convert.store_data_frame_stream', patch_store_data_frame_stream)
@patch('penelope.vendor.spacy.convert.load_data_frame_stream', patch_load_data_frame_stream)
def test_checkpoint_data_frame_succeeds():
    task = tasks.CheckpointDataFrame(pipeline=Mock(spec=CorpusPipeline), filename="dummy.zip").setup()
    task.instream = fake_data_frame_stream(1)
    for payload in task.outstream():
        assert payload.content_type == ContentType.DATAFRAME


def test_tokens_to_text_when_tokens_instream_succeeds():
    task = tasks.TokensToText(pipeline=Mock(spec=CorpusPipeline)).setup()
    current_payload = next(fake_token_stream(1))
    next_payload = task.process(current_payload)
    assert next_payload.content_type == ContentType.TEXT


def test_tokens_to_text_when_text_instream_succeeds():
    task = tasks.TokensToText(pipeline=Mock(spec=CorpusPipeline)).setup()
    current_payload = next(fake_text_stream(1))
    next_payload = task.process(current_payload)
    assert next_payload.content_type == ContentType.TEXT


def test_spacy_pipeline():
    checkpoint_filename = os.path.join(TEST_OUTPUT_FOLDER, "ssi_pos_csv.zip")
    text_reader_opts = TextReaderOpts(
        filename_fields=["document_id:_:2", "year:_:1"],
        filename_fields_key="document_id",
        filename_filter=None,
        filename_pattern="*.txt",
        as_binary=False,
    )

    pipeline_payload = PipelinePayload(
        source=TEST_CORPUS,
        document_index_source=None,
        pos_schema_name="Universal",
        memory_store={'spacy_model': "en_core_web_sm", 'nlp': None, 'lang': 'en,'},
    )
    pipeline = (
        SpacyPipeline(payload=pipeline_payload)
        .set_spacy_model(pipeline_payload.memory_store['spacy_model'])
        .load(reader_opts=text_reader_opts, transform_opts=TextTransformOpts())
        .text_to_spacy()
        .passthrough()
        .spacy_to_pos_dataframe()
        .checkpoint_dataframe(checkpoint_filename)
        .to_content()
    )

    df_docs = pipeline.resolve()
    assert next(df_docs) is not None
    assert os.path.isfile(checkpoint_filename)
    os.unlink(checkpoint_filename)
