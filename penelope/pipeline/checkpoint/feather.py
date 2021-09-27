import glob
import os
from os.path import join as jj
from typing import List, Optional

import pandas as pd
from penelope.utility import replace_extension, strip_paths

from ..interfaces import ContentType, DocumentPayload, PipelineError

FEATHER_DOCUMENT_INDEX_NAME = 'document_index.feathering'


def write_payload(folder: str, payload: DocumentPayload) -> DocumentPayload:

    filename: str = jj(folder, replace_extension(payload.filename, ".feather"))

    payload.content.reset_index(drop=True).to_feather(filename, compression="lz4")

    return payload


def payload_exists(folder: str, payload: DocumentPayload) -> DocumentPayload:
    filename = jj(folder, replace_extension(payload.filename, ".feather"))
    return os.path.isfile(filename)


def read_payload(filename: str) -> DocumentPayload:
    filename = replace_extension(filename, ".feather")
    return DocumentPayload(
        content_type=ContentType.TAGGED_FRAME,
        content=pd.read_feather(filename),
        filename=replace_extension(strip_paths(filename), ".csv"),
    )


def get_matching_paths(*, folder: str) -> List[str]:
    pattern: str = jj(folder, "*.feather")
    paths: List[str] = sorted(glob.glob(pattern))
    return paths


def document_index_exists(folder: Optional[str]) -> bool:
    if folder is None:
        return False
    return os.path.isfile(jj(folder, FEATHER_DOCUMENT_INDEX_NAME))


def read_document_index(folder: str) -> pd.DataFrame:

    filename = jj(folder, FEATHER_DOCUMENT_INDEX_NAME)

    if os.path.isfile(filename):
        document_index: pd.DataFrame = pd.read_feather(filename).set_index('document_name', drop=False)
        _sanitize_document_index(document_index)
        return document_index

    raise PipelineError("Feather checkpoint is missing document index. Please force new checkpoint!")


def write_document_index(folder: str, document_index: pd.DataFrame):

    if document_index is None:
        return

    _sanitize_document_index(document_index)
    filename = jj(folder, FEATHER_DOCUMENT_INDEX_NAME)
    document_index.reset_index(drop=True).to_feather(filename, compression="lz4")


def _sanitize_document_index(document_index: pd.DataFrame):

    if document_index is None:
        return

    if '' in document_index.columns:
        document_index.drop(columns='', inplace=True)

    if document_index.index.name in document_index.columns:
        document_index.rename_axis('', inplace=True)
