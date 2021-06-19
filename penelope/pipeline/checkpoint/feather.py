import glob
import os
from typing import Iterable, List

import pandas as pd
from penelope.corpus import DocumentIndex
from penelope.utility import replace_extension, strip_paths

from ..interfaces import ContentType, DocumentPayload, PipelineError
from ..tagged_frame import TaggedFrame

FEATHER_DOCUMENT_INDEX_NAME = 'document_index.feathering'


def write_payload(folder: str, payload: DocumentPayload) -> Iterable[DocumentPayload]:
    tagged_frame: TaggedFrame = payload.content
    filename = os.path.join(folder, replace_extension(payload.filename, ".feather"))
    tagged_frame.to_feather(filename, compression="lz4")
    return payload


def payload_exists(folder: str, payload: DocumentPayload) -> DocumentPayload:
    filename = os.path.join(folder, replace_extension(payload.filename, ".feather"))
    return os.path.isfile(filename)


def read_payload(filename: str) -> DocumentPayload:
    filename = replace_extension(filename, ".feather")
    tagged_frame: pd.DataFrame = pd.read_feather(filename)
    return DocumentPayload(
        content_type=ContentType.TAGGED_FRAME,
        content=tagged_frame,
        filename=replace_extension(strip_paths(filename), ".csv"),
    )


def get_matching_paths(*, folder: str) -> List[str]:
    pattern: str = os.path.join(folder, "*.feather")
    paths: List[str] = sorted(glob.glob(pattern))
    return paths


def document_index_exists(folder: str) -> bool:
    return os.path.isfile(os.path.join(folder, FEATHER_DOCUMENT_INDEX_NAME))


def read_document_index(folder: str) -> DocumentIndex:

    filename = os.path.join(folder, FEATHER_DOCUMENT_INDEX_NAME)

    if os.path.isfile(filename):
        document_index: DocumentIndex = pd.read_feather(filename).set_index('document_name', drop=False)
        if '' in document_index.columns:
            document_index.drop(columns='', inplace=True)
        return document_index

    raise PipelineError("Feather checkpoint is missing document index. Please force new checkpoint!")


def write_document_index(folder: str, document_index: DocumentIndex):
    filename = os.path.join(folder, FEATHER_DOCUMENT_INDEX_NAME)
    if document_index is not None:
        if document_index.index.name in document_index.columns:
            document_index.rename_axis('', inplace=True)
        document_index.reset_index().to_feather(filename, compression="lz4")
