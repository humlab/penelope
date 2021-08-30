from typing import List
from unittest.mock import MagicMock, Mock

import pandas as pd
from penelope.pipeline import checkpoint, sparv, interfaces, pipelines, tasks
from penelope.pipeline.tagged_frame import IngestVocabType, ToIdTaggedFrame
from penelope import utility
import pytest

TEST_CSV_POS_DOCUMENT: str = """token	pos	baseform
# text
Inne	AB	|inne|
i	RG	|
den	PN	|den|
väldiga	JJ	|väldig|
romanska	JJ	|romansk|
kyrkan	NN	|kyrka|
trängdes	VB	|tränga|trängas|
turisterna	NN	|turist|
i	PL	|
halvmörkret	NN	|halvmörker|
.	MAD	|
Valv	NN	|valv|
gapade	VB	|gapa|
bakom	PP	|bakom|
valv	NN	|valv|
och	UO	|
ingen	PN	|ingen|
överblick	NN	|överblick|
.	MAD	|
Några	DT	|någon|
ljuslågor	NN	|ljuslåga|
fladdrade	VB	|fladdra omkring:10|
.	MAD	|
"""


@pytest.mark.parametrize(
    'token_type, ingest_vocab_type, expected_tokens, expected_pos',
    [
        # (
        #     tasks.Vocabulary.TokenType.Lemma,
        #     IngestVocabType.Incremental,
        #     ['inne', 'i', 'den', 'väldig', 'någon', 'ljuslåga', 'fladdra_omkring', '.'],
        #     ['AB', 'RG', 'PN', 'JJ', 'DT', 'NN', 'VB', 'MAD'],
        # ),
        (
            tasks.Vocabulary.TokenType.Lemma,
            IngestVocabType.Prebuild,
            ['inne', 'i', 'den', 'väldig', 'någon', 'ljuslåga', 'fladdra_omkring', '.'],
            ['AB', 'RG', 'PN', 'JJ', 'DT', 'NN', 'VB', 'MAD'],
        ),
        # (
        #     tasks.Vocabulary.TokenType.Text,
        #     IngestVocabType.Incremental,
        #     ['inne', 'i', 'den', 'väldig', 'någon', 'ljuslåga', 'fladdra_omkring', '.'],
        #     ['AB', 'RG', 'PN', 'JJ', 'DT', 'NN', 'VB', 'MAD'],
        # ),
    ],
)

def test_id_tagged_frame_process_payload(
    token_type: tasks.Vocabulary.TokenType,
    ingest_vocab_type: IngestVocabType,
    expected_tokens: List[str],
    expected_pos: List[str],
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
        },
    )

    tagged_frame: pd.DataFrame = sparv.SparvCsvSerializer().deserialize(
        TEST_CSV_POS_DOCUMENT,
        checkpoint.CheckpointOpts(),
    )

    payload = interfaces.DocumentPayload(content_type=interfaces.ContentType.TAGGED_FRAME, content=tagged_frame)
    prior = MagicMock(spec=interfaces.ITask, outstream=lambda: [payload])

    task: ToIdTaggedFrame = ToIdTaggedFrame(
        pipeline=pipeline,
        prior=prior,
        token_type=token_type,
        close=True,
        ingest_vocab_type=ingest_vocab_type,
        tf_keeps=set(),
        tf_threshold=None,
    ).setup()

    task.enter()
    next_payload: interfaces.DocumentPayload = task.process_payload(payload)

    assert next_payload is not None
    assert next_payload.content_type == interfaces.ContentType.TAGGED_ID_FRAME

    tagged_id_frame: pd.DataFrame = next_payload.content

    assert 'token_id' in tagged_id_frame.columns
    assert 'pos_id' in tagged_id_frame.columns

    assert not tagged_id_frame.token_id.isna().any()
    assert not tagged_id_frame.pos_id.isna().any()

    tokens: List[str] = tagged_id_frame.token_id.map(task.token2id.id2token).tolist()
    assert tokens[:4] + tokens[-4:] == expected_tokens

    pos: List[str] = tagged_id_frame.pos_id.map(utility.PoS_TAGS_SCHEMES.SUC.id_to_pos).tolist()
    assert pos[:4] + pos[-4:] == expected_pos
