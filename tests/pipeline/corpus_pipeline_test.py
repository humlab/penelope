import csv
import os
import pathlib
from typing import Iterable, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import penelope.pipeline.spacy.tasks as spacy_tasks
import penelope.pipeline.tasks as tasks
import pytest
import spacy.language
import spacy.tokens
from penelope.corpus import load_document_index
from penelope.corpus.readers import ExtractTaggedTokensOpts, TextReaderOpts, TextTransformOpts
from penelope.pipeline import (
    CheckpointData,
    CheckpointOpts,
    ContentType,
    CorpusConfig,
    CorpusPipeline,
    DocumentPayload,
    PipelinePayload,
)
from penelope.utility import PropertyValueMaskingOpts
from tests.utils import TEST_DATA_FOLDER

# pylint: disable=redefined-outer-name

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


def monkey_patch(*_, **__):
    return Mock()


@pytest.fixture(scope="module")
def mary_had_a_little_lamb_corpus() -> Iterable[Tuple[str, str]]:
    return TEST_CORPUS


@pytest.fixture(scope="module")
def reader_opts():
    return TextReaderOpts(
        filename_fields=["file_id:_:2", "year:_:1"],
        index_field=None,
        filename_filter=None,
        filename_pattern="*.txt",
        as_binary=False,
    )


def fake_data_frame_stream(n: int = 1):
    for i in range(1, n + 1):
        yield DocumentPayload(
            filename=f'dummy_{i}.csv',
            content_type=ContentType.TAGGED_FRAME,
            content=pd.DataFrame(data={'text': ['bil'], 'pos_': ['NOUN'], 'lemma_': ['bil']}),
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


@pytest.fixture(scope="module")
def checkpoint_opts() -> CheckpointOpts:
    opts = CheckpointOpts(
        content_type_code=1,
        document_index_name='document_index.csv',
        document_index_sep='\t',
        index_column=0,
        text_column='text',
        lemma_column='lemma_',
        pos_column='pos_',
        sep='\t',
        quoting=csv.QUOTE_NONE,
    )
    return opts


def patch_spacy_doc(*x, **y):  # pylint: disable=unused-argument
    m = MagicMock(spec=spacy.tokens.Doc)
    return m


def fake_config():
    return Mock(spec=CorpusConfig, pipeline_payload=PipelinePayload(memory_store=dict(spacy_nlp=patch_spacy_load())))


def patch_spacy_pipeline(task):
    pipeline = CorpusPipeline(config=fake_config(), tasks=[task]).setup()
    return pipeline


@patch('spacy.load', patch_spacy_load)
def test_set_spacy_model_setup_succeeds():
    pipeline = CorpusPipeline(config=fake_config())
    _ = spacy_tasks.SetSpacyModel(pipeline=pipeline, lang_or_nlp="en_core_web_sm").setup()
    assert pipeline.get("spacy_nlp", None) is not None


def test_load_text_when_source_is_list_of_filename_text_tuples_succeeds(
    reader_opts,
):  # pylint: disable=redefined-outer-name
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
    task = tasks.Passthrough(
        pipeline=CorpusPipeline(config=Mock(spec=CorpusConfig, pipeline_payload=PipelinePayload()))
    ).setup()
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
    task = spacy_tasks.ToSpacyDoc(pipeline=Mock(spec=CorpusPipeline)).setup()
    current_payload = next(fake_text_stream())
    next_payload = task.process(current_payload)
    assert next_payload.content_type == ContentType.SPACYDOC


def test_text_to_spacy_process_with_non_text_payload_fails():
    task = spacy_tasks.ToSpacyDoc(pipeline=Mock(spec=CorpusPipeline)).setup()
    current_payload = next(fake_data_frame_stream(1))
    with pytest.raises(Exception) as _:
        _ = task.setup().process(current_payload)


def patch_any_to_tagged_frame(
    spacy_doc, attributes, attribute_value_filters  # pylint: disable=unused-argument
) -> pd.DataFrame:
    return pd.DataFrame(data={'text': ['bil'], 'pos_': ['NOUN'], 'lemma_': ['bil']})


def patch_spacy_doc_to_tagged_frame(
    spacy_doc, attributes, attribute_value_filters  # pylint: disable=unused-argument
) -> pd.DataFrame:
    return pd.DataFrame(data={'text': ['bil'], 'pos_': ['NOUN'], 'lemma_': ['bil']})


@patch('penelope.pipeline.spacy.convert.spacy_doc_to_tagged_frame', patch_spacy_doc_to_tagged_frame)
def test_text_to_tagged_frame_with_text_payload_succeeds():
    task = spacy_tasks.ToSpacyDocToTaggedFrame(
        pipeline=Mock(spec=CorpusPipeline),
    ).setup()
    task.tagger = MagicMock(name='tagger')
    task.register_token_counts = MagicMock(name='register_token_counts')
    current_payload = next(fake_text_stream())
    next_payload = task.process(current_payload)
    assert task.tagger.call_count == 1
    assert task.register_token_counts.call_count == 1
    assert next_payload.content_type == ContentType.TAGGED_FRAME


@patch('penelope.pipeline.spacy.convert.spacy_doc_to_tagged_frame', patch_any_to_tagged_frame)
def test_spacy_to_tagged_frame_with_doc_payload_succeeds():
    task = spacy_tasks.SpacyDocToTaggedFrame(pipeline=Mock(spec=CorpusPipeline)).setup()
    task.tagger = MagicMock(name='tagger')
    task.register_token_counts = MagicMock(name='register_token_counts')
    current_payload = next(fake_spacy_doc_stream())
    next_payload = task.process(current_payload)
    assert task.tagger.call_count == 1
    assert task.register_token_counts.call_count == 1
    assert next_payload.content_type == ContentType.TAGGED_FRAME


def patch_tagged_frame_to_tokens(*_, **__) -> Iterable[str]:
    return ["a", "b", "c"]


@patch('penelope.pipeline.convert.tagged_frame_to_tokens', patch_tagged_frame_to_tokens)
def test_tagged_frame_to_tokens_succeeds():
    pipeline = Mock(spec=CorpusPipeline, payload=Mock(spec=PipelinePayload, tagged_columns_names={}))
    task = tasks.TaggedFrameToTokens(
        pipeline=pipeline,
        extract_opts=ExtractTaggedTokensOpts(lemmatize=True),
        filter_opts=PropertyValueMaskingOpts(is_punct=False),
    ).setup()
    current_payload = next(fake_data_frame_stream(1))
    next_payload = task.process(current_payload)
    assert next_payload.content_type == ContentType.TOKENS


def patch_store_checkpoint(
    *, checkpoint_opts, target_filename, document_index, payload_stream  # pylint: disable=unused-argument
):
    for p in payload_stream:
        yield p


def patch_load_checkpoint(*_, **__) -> Tuple[Iterable[DocumentPayload], Optional[pd.DataFrame]]:
    return CheckpointData(
        content_type=ContentType.TAGGED_FRAME,
        document_index=None,
        create_stream=lambda: fake_data_frame_stream(1),
        checkpoint_opts=CheckpointOpts().as_type(ContentType.TAGGED_FRAME),
        source_name="source-name",
    )


@patch('penelope.pipeline.checkpoint.store_checkpoint', patch_store_checkpoint)
def test_save_data_frame_succeeds():
    pipeline = Mock(spec=CorpusPipeline, **{'payload.set_reader_index': monkey_patch})
    opts = Mock(spec=CheckpointOpts)
    task = tasks.SaveTaggedCSV(pipeline=pipeline, filename="dummy.zip", checkpoint_opts=opts)
    task.instream = fake_data_frame_stream(1)
    for payload in task.outstream():
        assert payload.content_type == ContentType.TAGGED_FRAME


@patch('penelope.pipeline.checkpoint.load_checkpoint', patch_load_checkpoint)
def test_load_data_frame_succeeds():
    pipeline = Mock(
        spec=CorpusPipeline,
        **{
            'payload.set_reader_index': monkey_patch,
        },
    )
    task = tasks.LoadTaggedCSV(pipeline=pipeline, filename="dummy.zip", extra_reader_opts=TextReaderOpts()).setup()
    task.register_token_counts = lambda _: task
    task.instream = fake_data_frame_stream(1)
    for payload in task.outstream():
        assert payload.content_type == ContentType.TAGGED_FRAME


@patch('penelope.pipeline.checkpoint.store_checkpoint', patch_store_checkpoint)
@patch('penelope.pipeline.checkpoint.load_checkpoint', patch_load_checkpoint)
def test_checkpoint_data_frame_succeeds():
    attrs = {'get_prior_content_type.return_value': ContentType.TAGGED_FRAME}
    task = tasks.Checkpoint(pipeline=Mock(spec=CorpusPipeline, **attrs), filename="dummy.zip").setup()
    task.instream = fake_data_frame_stream(1)
    for payload in task.outstream():
        assert payload.content_type == ContentType.TAGGED_FRAME


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


def test_spacy_pipeline(checkpoint_opts: CheckpointOpts):

    checkpoint_filename = os.path.join(TEST_OUTPUT_FOLDER, "checkpoint_mary_lamb_pos_csv.zip")

    pathlib.Path(checkpoint_filename).unlink(missing_ok=True)

    text_reader_opts = TextReaderOpts(
        filename_fields=["doc_id:_:2", "year:_:1"],
        index_field=None,  # use filename
        filename_filter=None,
        filename_pattern="*.txt",
        as_binary=False,
    )

    pipeline_payload = PipelinePayload(
        source=TEST_CORPUS,
        document_index_source=None,
        pos_schema_name="Universal",
        memory_store={'spacy_model': "en_core_web_sm", 'nlp': None, 'lang': 'en,', 'pos_column': 'pos_'},
    )
    config = Mock(spec=CorpusConfig, pipeline_payload=pipeline_payload)
    pipeline = (
        CorpusPipeline(config=config)
        .set_spacy_model(pipeline_payload.memory_store['spacy_model'])
        .load_text(reader_opts=text_reader_opts, transform_opts=TextTransformOpts())
        .text_to_spacy()
        .passthrough()
        .spacy_to_pos_tagged_frame()
        .checkpoint(checkpoint_filename, checkpoint_opts=checkpoint_opts)
        .to_content()
    )

    df_docs = pipeline.resolve()
    assert next(df_docs) is not None
    assert os.path.isfile(checkpoint_filename)

    # pathlib.Path(checkpoint_filename).unlink(missing_ok=True)


def test_spacy_pipeline_load_checkpoint(checkpoint_opts: CheckpointOpts):

    checkpoint_filename = os.path.join(TEST_DATA_FOLDER, "checkpoint_mary_lamb_pos_csv.zip")

    pipeline_payload = PipelinePayload(
        source=TEST_CORPUS,
        document_index_source=None,
        pos_schema_name="Universal",
        memory_store={'spacy_model': "en_core_web_sm", 'nlp': None, 'lang': 'en,'},
    )
    config = MagicMock(spec=CorpusConfig, payload=pipeline_payload)
    pipeline = (
        CorpusPipeline(config=config)
        .checkpoint(
            checkpoint_filename,
            checkpoint_opts=checkpoint_opts,
        )
        .to_content()
    )

    df_docs = pipeline.resolve()
    assert next(df_docs) is not None


def test_load_primary_document_index():

    filename = './tests/test_data/legal_instrument_five_docs_test.csv'
    df = load_document_index(filename, sep=';')

    assert df is not None
    assert 'unesco_id' in df.columns
    assert 'document_id' in df.columns


def store_document_index(document_index: pd.DataFrame, filename: str):
    document_index.to_csv(filename, sep='\t', header=True, index=True)
