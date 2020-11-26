import zipfile
from io import StringIO
from typing import Iterable, Iterator, Union

import pandas as pd
from penelope.utility import replace_extension

from .interfaces import ContentType, DocumentPayload


def to_text(data: Union[str, Iterable[str]]):
    return data if isinstance(data, str) else ' '.join(data)


def read_data_frame_from_zip(zf, filename):
    data_str = zf.read(filename).decode('utf-8')
    data_source = StringIO(data_str)
    df = pd.read_csv(data_source, sep='\t', index_col=0)
    return df


def write_data_frame_to_zip(df: pd.DataFrame, filename: str, zf: zipfile.ZipFile):
    assert isinstance(df, (pd.DataFrame,))
    data_str: str = df.to_csv(sep='\t', header=True)
    zf.writestr(filename, data=data_str)


def store_data_frame_outstream(
    target_filename: str, document_index: pd.DataFrame, payload_stream: Iterator[DocumentPayload]
):
    with zipfile.ZipFile(target_filename, mode="w", compresslevel=zipfile.ZIP_DEFLATED) as zf:
        write_data_frame_to_zip(document_index, "document_index.csv", zf)
        for payload in payload_stream:
            filename = replace_extension(payload.filename, ".csv")
            write_data_frame_to_zip(payload.content, filename, zf)
            yield payload


def load_data_frame_instream(self, source_filename: str) -> Iterable[DocumentPayload]:
    document_index_name = self.payload.document_index_filename
    self.payload.source = source_filename
    with zipfile.ZipFile(source_filename, mode="r") as zf:
        filenames = zf.namelist()
        if document_index_name in filenames:
            self.payload.document_index = read_data_frame_from_zip(zf, document_index_name)
            filenames.remove(document_index_name)
        for filename in filenames:
            df = read_data_frame_from_zip(zf, filename)
            payload = DocumentPayload(content_type=ContentType.DATAFRAME, content=df, filename=filename)
            yield payload


def consolidate_document_index(index_filename: str, index:pd.DataFrame, reader_index: pd.DataFrame):
    """Returns a consolidated document index from ax existing index, if exists,
    and index created by reader. If index is not loaded, and no filename is given,
    then the reader index is returned. Otherwise the two indexes are merged."""

    if index is None:
        if index_filename:
            index = pd.read_csv(index_filename, sep='\t', index_col=0)

    if index is not None:
        columns = [x for x in reader_index.columns if x not in index.columns]
        index = index.merge(reader_index[columns], left_index=True, right_index=True, how='left')
    else:
        index = reader_index

    return index
