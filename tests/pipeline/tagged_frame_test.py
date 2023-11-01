from typing import List
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

from penelope import pipeline as pp
from penelope import utility
from penelope.corpus.serialize import SerializeOpts
from penelope.pipeline import interfaces, pipelines, sparv, tasks
from penelope.pipeline.tagged_frame import IngestVocabType, ToIdTaggedFrame

from ..fixtures import TEST_CSV_POS_DOCUMENT


@pytest.mark.parametrize(
    'lowercase, token_type, ingest_vocab_type, expected_tokens, expected_pos, expected_vocab_count',
    [
        (
            False,
            tasks.Vocabulary.TokenType.Lemma,
            IngestVocabType.Incremental,
            'inne i den väldig någon ljuslåga fladdra_omkring .',
            ['AB', 'RG', 'PN', 'JJ', 'DT', 'NN', 'VB', 'MAD'],
            21,
        ),
        (
            False,
            tasks.Vocabulary.TokenType.Lemma,
            IngestVocabType.Prebuild,
            'inne i den väldig någon ljuslåga fladdra_omkring .',
            ['AB', 'RG', 'PN', 'JJ', 'DT', 'NN', 'VB', 'MAD'],
            21,
        ),
        (
            False,
            tasks.Vocabulary.TokenType.Text,
            IngestVocabType.Incremental,
            'Inne i den väldiga Några ljuslågor fladdrade .',
            ['AB', 'RG', 'PN', 'JJ', 'DT', 'NN', 'VB', 'MAD'],
            22,
        ),
        (
            True,
            tasks.Vocabulary.TokenType.Text,
            IngestVocabType.Incremental,
            'Inne i den väldiga Några ljuslågor fladdrade .',
            ['AB', 'RG', 'PN', 'JJ', 'DT', 'NN', 'VB', 'MAD'],
            22,
        ),
    ],
)
def test_id_tagged_frame_process_payload(
    lowercase: bool,
    token_type: tasks.Vocabulary.TokenType,
    ingest_vocab_type: IngestVocabType,
    expected_tokens: str,
    expected_pos: List[str],
    expected_vocab_count: int,
):
    memory_store = {
        'text_column': 'token',
        'lemma_column': 'baseform',
        'pos_column': 'pos',
    }

    pipeline = Mock(
        spec=pipelines.CorpusPipeline,
        **{
            'config.pipeline_payload.pos_schema': utility.PoS_TAGS_SCHEMES.SUC,
            'payload.memory_store': memory_store,
            'get': lambda key, _: memory_store[key],
            'payload.document_index.index': [None],
        },
    )

    tagged_frame: pd.DataFrame = sparv.SparvCsvSerializer().deserialize(
        content=TEST_CSV_POS_DOCUMENT,
        options=SerializeOpts(**memory_store),
    )

    payload = interfaces.DocumentPayload(content_type=interfaces.ContentType.TAGGED_FRAME, content=tagged_frame)
    prior = MagicMock(spec=interfaces.ITask, outstream=lambda **_: [payload])

    task: ToIdTaggedFrame = ToIdTaggedFrame(
        pipeline=pipeline,
        prior=prior,
        lowercase=lowercase,
        token_type=token_type,
        close=True,
        ingest_vocab_type=ingest_vocab_type,
        tf_keeps=set(),
        tf_threshold=None,
    ).setup()

    task.enter()

    assert task.token2id is not None

    if ingest_vocab_type == IngestVocabType.Prebuild:
        assert len(task.token2id) == expected_vocab_count

    next_payload: interfaces.DocumentPayload = task.process_payload(payload)

    if ingest_vocab_type == IngestVocabType.Incremental:
        assert len(task.token2id) == expected_vocab_count

    assert next_payload is not None
    assert next_payload.content_type == interfaces.ContentType.TAGGED_ID_FRAME

    tagged_id_frame: pd.DataFrame = next_payload.content

    assert 'token_id' in tagged_id_frame.columns
    assert 'pos_id' in tagged_id_frame.columns

    assert not tagged_id_frame.token_id.isna().any()
    assert not tagged_id_frame.pos_id.isna().any()

    tokens: List[str] = tagged_id_frame.token_id.map(task.token2id.id2token).tolist()
    assert tokens[:4] + tokens[-4:] == expected_tokens.split()

    pos: List[str] = tagged_id_frame.pos_id.map(utility.PoS_TAGS_SCHEMES.SUC.id_to_pos).tolist()
    assert pos[:4] + pos[-4:] == expected_pos


@pytest.mark.long_running
@pytest.mark.skip(reason="only used to create test data")
def test_store_id_tagged_frame():
    config: pp.CorpusConfig = pp.CorpusConfig.load('./tests/test_data/tranströmer/tranströmer.yml')
    corpus_source: str = './tests/test_data/tranströmer/tranströmer_corpus_pos_csv.zip'
    target_folder: str = './tests/test_data/tranströmer/tranströmer_id_tagged_frames'
    _: pp.CorpusPipeline = (
        pp.CorpusPipeline(config=config)
        .load_tagged_frame(
            filename=corpus_source,
            serialize_opts=config.serialize_opts,
            extra_reader_opts=config.text_reader_opts,
        )
        .to_id_tagged_frame(ingest_vocab_type=IngestVocabType.Incremental)
        .store_id_tagged_frame(target_folder)
    ).exhaust()


def test_load_id_tagged_frame():
    config: pp.CorpusConfig = pp.CorpusConfig.load('./tests/test_data/tranströmer/tranströmer.yml')
    folder: str = './tests/test_data/tranströmer/tranströmer_id_tagged_frames'
    p: pp.CorpusPipeline = pp.CorpusPipeline(config=config).load_id_tagged_frame(
        folder=folder,
        file_pattern='**/tran*.feather',
        id_to_token=False,
    )

    payloads = p.to_list()

    assert len(payloads) == 5
    assert len(p.payload.document_index) == 5
    assert len(p.payload.token2id) == 341
