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


# def test_deserialize_prot_198384__34():

#     filename = 'tests/test_data/prot_198384__34.csv'
#     with open(filename, "r") as fp:
#         content: str = fp.read()

#     serializer_cls: Type[sparv.SparvCsvSerializer] = utility.create_instance(
#         'penelope.pipeline.sparv.SparvCsvSerializer'
#     )
#     serializer: sparv.SparvCsvSerializer = serializer_cls()
#     options: checkpoint.CheckpointOpts = checkpoint.CheckpointOpts(
#         content_type_code = 1,
#         document_index_name = None,
#         document_index_sep = None,
#         sep = '\t',
#         quoting = 3,
#         custom_serializer_classname = "penelope.pipeline.sparv.convert.SparvCsvSerializer",
#         deserialize_in_parallel = False,
#         deserialize_processes = 4,
#         deserialize_chunksize = 4,
#         text_column = "token",
#         lemma_column = "baseform",
#         pos_column = "pos",
#         extra_columns = [],
#         index_column = None
#     )

#     tagged_frame: pd.DataFrame = serializer.deserialize(content, options)

#     assert tagged_frame is not None
#     assert tagged_frame.token.tolist()[:5] == ['RIKSDAGENS', 'PROTOKOLL', '1950', 'ANDRA', 'KAMMAREN']
#     # FIXME: #32 Verify quality of Sparv4 CSV dehyphenation
#     assert tagged_frame.token.tolist()[-5:] == ['bandelen', 'Öster-', 'sund', '—', 'Gällivare']
#     assert tagged_frame.baseform.tolist()[:5] == ['riksdag', 'protokoll', '1950', 'andra', 'kammare']
#     assert tagged_frame.baseform.tolist()[-5:] == ['bandel', 'Öster-', 'sund', '—', 'Gällivare']
#     assert all(~tagged_frame.token.isna())
#     assert all(~tagged_frame.baseform.isna())
#     assert all(~tagged_frame.pos.isna())


