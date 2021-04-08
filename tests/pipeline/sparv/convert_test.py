from typing import Type

import pandas as pd
from penelope import utility
from penelope.pipeline import checkpoint, sparv


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
    # FIXME: #32 Verify quality of Sparv4 CSV dehyphenation
    assert tagged_frame.token.tolist()[-5:] == ['bandelen', 'Öster-', 'sund', '—', 'Gällivare']
    assert tagged_frame.baseform.tolist()[:5] == ['riksdag', 'protokoll', '1950', 'andra', 'kammare']
    assert tagged_frame.baseform.tolist()[-5:] == ['bandel', 'Öster-', 'sund', '—', 'Gällivare']
    assert all(~tagged_frame.token.isna())
    assert all(~tagged_frame.baseform.isna())
    assert all(~tagged_frame.pos.isna())
