from typing import Iterable
from unittest.mock import MagicMock, Mock

import pandas as pd
from penelope.pipeline import checkpoint, sparv
from penelope.pipeline.interfaces import ContentType, DocumentPayload
from penelope.pipeline.pipelines import CorpusPipeline
from penelope.pipeline.tasks import Vocabulary


def test_task_vocabulary_token2id():

    filename = 'tests/test_data/riksdagens-protokoll.test.sparv4.csv'

    with open(filename, "r") as fp:
        content: str = fp.read()

    pipeline = Mock(
        spec=CorpusPipeline,
        **{
            'payload.memory_store': {'text_column': 'token', 'lemma_column': 'baseform'},
        },
    )

    instream = MagicMock(spec=Iterable[DocumentPayload])
    task: Vocabulary = Vocabulary(pipeline=pipeline, instream=instream).setup()

    tagged_frame: pd.DataFrame = sparv.SparvCsvSerializer().deserialize(
        content,
        checkpoint.CheckpointOpts(),
    )

    payload = DocumentPayload(content_type=ContentType.TAGGED_FRAME, content=tagged_frame)

    expected_tokens = tagged_frame.token.tolist() + tagged_frame.baseform.tolist()
    assert expected_tokens == [x for x in task.tokens_iter(payload)]

    payload_next = task.process_payload(payload=payload)

    assert payload_next is not None
    assert payload_next.content_type == ContentType.TAGGED_FRAME
    assert payload_next.content is tagged_frame
    assert len(set(expected_tokens)) == len(task.token2id)
    assert task.token2id is pipeline.payload.token2id
