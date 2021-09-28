import pandas as pd
import pytest
from penelope import pipeline


def test_bug():

    serializer: pipeline.CsvContentSerializer = pipeline.CsvContentSerializer()

    with open("./tests/test_data/1955_069029_026.csv", "r") as fp:
        content: str = fp.read()

    serialize_opts: pipeline.CheckpointOpts = pipeline.CheckpointOpts(
        content_type_code=1,
        sep="\t",
        quoting=3,
        document_index_name=None,
        document_index_sep="\t",
        text_column='text',
        lemma_column='lemma_',
        pos_column='pos_',
        extra_columns=[],
        custom_serializer_classname=None,
        deserialize_processes=2,
        deserialize_chunksize=4,
        feather_folder='/data/inidun/courier_page_20210921.feather',
        index_column=None,
    )

    tagged_frame: pd.DataFrame = serializer.deserialize(content=content, options=serialize_opts)

    _ = tagged_frame.lemma_.str.lower()

    tagged_frame = pd.read_feather('./tests/test_data/1955_069029_026.feather')
    assert tagged_frame is not None


@pytest.mark.skip("APA")
def test_read_from_zip():
    zip_or_filename = '/data/inidun/courier_page_20210921_pos_csv.zip'
    filename = '1955_069029_029.txt'
    checkpoint_opts = pipeline.CheckpointOpts(
        content_type_code=1,
        sep="\t",
        quoting=3,
        document_index_name=None,
        document_index_sep="\t",
        text_column='text',
        lemma_column='lemma_',
        pos_column='pos_',
        extra_columns=[],
        custom_serializer_classname=None,
        deserialize_processes=2,
        deserialize_chunksize=4,
        feather_folder='/data',
    )
    serializer = pipeline.CsvContentSerializer()
    payload = pipeline.checkpoint.load_payload(
        zip_or_filename=zip_or_filename,
        filename=filename,
        checkpoint_opts=checkpoint_opts,
        serializer=serializer,
    )

    assert payload is not None
