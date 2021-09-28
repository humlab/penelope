import os
from typing import Iterable
from unittest.mock import MagicMock, Mock

import pandas as pd
from penelope.corpus.document_index import load_document_index_from_str
from penelope.pipeline import checkpoint, sparv
from penelope.pipeline.checkpoint import feather
from penelope.pipeline.interfaces import ContentType, DocumentPayload, ITask
from penelope.pipeline.pipelines import CorpusPipeline
from penelope.pipeline.tasks import Vocabulary
from tests.pipeline.fixtures import SPARV_TAGGED_COLUMNS


def test_task_vocabulary_token2id():

    filename = 'tests/test_data/riksdagens-protokoll.test.sparv4.csv'

    with open(filename, "r") as fp:
        content: str = fp.read()

    pipeline = Mock(
        spec=CorpusPipeline,
        **{
            'payload.memory_store': SPARV_TAGGED_COLUMNS,
        },
    )

    prior = MagicMock(spec=ITask, outstream=lambda: MagicMock(spec=Iterable[DocumentPayload]))

    task: Vocabulary = Vocabulary(pipeline=pipeline, prior=prior, token_type=Vocabulary.TokenType.Lemma).setup()

    tagged_frame: pd.DataFrame = sparv.SparvCsvSerializer().deserialize(
        content=content,
        options=checkpoint.CheckpointOpts(),
    )

    payload = DocumentPayload(content_type=ContentType.TAGGED_FRAME, content=tagged_frame)

    expected_tokens = tagged_frame.baseform.str.lower().tolist()
    assert expected_tokens == [x for x in task.tokens_stream(payload)]


def test_CheckpointFeather_write_document_index():
    folder = './tests/output'
    expected_filename = os.path.join(folder, feather.FEATHER_DOCUMENT_INDEX_NAME)

    os.makedirs(folder, exist_ok=True)

    data_str = """
filename;year;year_id;document_name;document_id;title;n_raw_tokens
A.txt;2019;1;A;0;Even break;68
B.txt;2019;2;B;1;Night;59

"""
    document_index = load_document_index_from_str(data_str, sep=';')

    feather.write_document_index(folder, document_index)

    assert os.path.isfile(expected_filename)

    df = feather.read_document_index(folder)

    assert df is not None
