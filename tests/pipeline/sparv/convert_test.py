from typing import Type

import pandas as pd
from penelope import utility
from penelope.pipeline import CheckpointOpts, CsvContentSerializer, checkpoint, sparv
from penelope.pipeline.sparv import deserialize_lemma_form


def test_sparv_csv_serializer():

    filename = 'tests/test_data/riksdagens-protokoll.test.sparv4.csv'
    with open(filename, "r") as fp:
        content: str = fp.read()

    serializer_cls: Type[sparv.SparvCsvSerializer] = utility.create_instance(
        'penelope.pipeline.sparv.SparvCsvSerializer'
    )
    serializer: sparv.SparvCsvSerializer = serializer_cls()
    options: checkpoint.CheckpointOpts = checkpoint.CheckpointOpts()

    tagged_frame: pd.DataFrame = serializer.deserialize(content, options)

    assert tagged_frame is not None
    assert tagged_frame.token.tolist()[:5] == ['RIKSDAGENS', 'PROTOKOLL', '1950', 'ANDRA', 'KAMMAREN']
    assert tagged_frame.token.tolist()[-5:] == ['sund', '—', 'Gällivare', 'Roger', 'Mähler']
    assert tagged_frame.baseform.tolist()[:5] == ['riksdag', 'protokoll', '1950', 'andra', 'kammare']
    assert tagged_frame.baseform.tolist()[-5:] == ['sund', '—', 'Gällivare', 'Roger', 'super_man']
    assert all(~tagged_frame.token.isna())
    assert all(~tagged_frame.baseform.isna())
    assert all(~tagged_frame.pos.isna())


def test_sparv_csv_deserialize_lemma_form():

    tagged_frame_str: str = """token	pos	baseform
# text
RIKSDAGEN	NN	|riksdag|
1975	RG	|
/	PAD	|
76	RG	|
Protokoll	NN	|protokoll|
Nr	NN	|nr|
1	RG	|
—	MID	|
15	RG	|
15	RG	|
oktober	NN	|oktober|
—	MID	|
4	RG	|
november	NN	|november|
Band	PM	|
A	IN	|
1	RG	|"""

    serializer: CsvContentSerializer = CsvContentSerializer()
    checkpoint_opts: CheckpointOpts = CheckpointOpts(
        text_column='token',
        lemma_column='baseform',
        pos_column='pos',
        index_column=None,
        sep='\t',
    )
    tagged_frame: pd.DataFrame = serializer.deserialize(tagged_frame_str, checkpoint_opts)

    assert tagged_frame is not None

    lemma: pd.Series = deserialize_lemma_form(tagged_frame)

    assert lemma is not None
