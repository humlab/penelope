from typing import Iterable, Type
from unittest.mock import MagicMock, Mock

import pandas as pd
from penelope import utility
from penelope.pipeline import config, sparv
from penelope.pipeline.interfaces import ContentType, DocumentPayload
from penelope.pipeline.pipelines import CorpusPipeline
from penelope.pipeline.tasks import Vocabulary


def create_test_content():
    content = """token	pos	baseform
# text
RIKSDAGENS	NN	|riksdag|
PROTOKOLL	NN	|protokoll|
1950	RG	|
ANDRA	JJ	|andra|
KAMMAREN	NN	|kammare|
Nr	NN	|nr|
1	RG	|
10	RG	|
—	MID	|
17	RG	|
januari	NN	|januari|
.	MAD	|
Debatter	NN	|debatt|
m	NN	|m|
.	MAD	|
m	NN	|m|
.	MAD	|
Tisdagen	NN	|tisdag|
den	PN	|den|
10	RG	|
januari	NN	|januari|
.	MAD	|
Ålderspresidentens	NN	|ålderspresident|
hälsningstal	NN	|hälsningstal|
Onsdagen	NN	|onsdag|
den	PN	|den|
11	RG	|
januari	NN	|januari|
.	MAD	|
Talmannens	NN	|talman|
anförande	NN	|anförande|
vid	PP	|vid|
riksdagens	NN	|riksdag|
öppnande	PC	|öppna|
Torsdagen	NN	|torsdag|
den	PN	|den|
12	RG	|
januari	NN	|januari|
.	MAD	|
Interpellationer	NN	|interpellation|
av	PL	|av|
:	MID	|
Herr	NN	|herr|
Jacobson	PM	|Jacobson|
i	RG	|
Vilhelmina	PM	|Vilhelmina|
ang.	AB	|
persontrafiken	NN	|persontrafik|
å	KN	|
bandelen	NN	|bandel|
Öster-	JJ	|
sund	NN	|sund|
—	MID	|
Gällivare	PM	|Gällivare|"""
    # zip_or_filename: str = './tests/test_data/riksdagens-protokoll.test.2smallfiles.zip'
    # document_name: str = 'prot_1950__ak__1.csv'
    # content: str = utility.zip_utils.read(zip_or_filename=zip_or_filename, filename=document_name, as_binary=False)
    return content


def test_sparv_csv_serializer():

    serializer_cls: Type[sparv.SparvCsvSerializer] = utility.create_instance(
        'penelope.pipeline.sparv.SparvCsvSerializer'
    )
    serializer: sparv.SparvCsvSerializer = serializer_cls()
    options: config.CheckpointSerializeOpts = config.CheckpointSerializeOpts()
    content: str = create_test_content()

    tagged_frame: pd.DataFrame = serializer.deserialize(content, options)

    assert tagged_frame is not None
    assert tagged_frame.token.tolist()[:5] == ['RIKSDAGENS', 'PROTOKOLL', '1950', 'ANDRA', 'KAMMAREN']
    # FIXME: #32 Verify quality of Sparv4 CSV dehyphenation
    assert tagged_frame.token.tolist()[-5:] == ['bandelen', 'Öster-', 'sund', '—', 'Gällivare']
    assert tagged_frame.baseform.tolist()[:5] == ['riksdag', 'protokoll', '1950', 'andra', 'kammare']
    assert tagged_frame.baseform.tolist()[-5:] == ['bandel', 'Öster-', 'sund', '—', 'Gällivare']
    assert all(~tagged_frame.token.isna())
    assert all(~tagged_frame.baseform.isna())
    assert all(~tagged_frame.pos.isna())


def test_sparv_csv_create_token2id():

    pipeline = Mock(
        spec=CorpusPipeline, **{'payload.tagged_columns_names': {'text_column': 'token', 'lemma_column': 'baseform'}}
    )
    instream = MagicMock(spec=Iterable[DocumentPayload])
    task: Vocabulary = Vocabulary(pipeline=pipeline, instream=instream).setup()

    tagged_frame: pd.DataFrame = sparv.SparvCsvSerializer().deserialize(
        create_test_content(),
        config.CheckpointSerializeOpts(),
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
