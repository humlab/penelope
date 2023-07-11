import glob
import os
from os.path import basename, isfile
from os.path import join as jj

import pandas as pd

from penelope.utility import replace_extension, strip_paths
from penelope.utility.filename_utils import strip_path_and_extension

from ..interfaces import ContentType, DocumentPayload


def get_document_filenames(*, folder: str) -> list[str]:
    """Returns list of document filenames in folder"""
    return [
        x
        for x in sorted(glob.glob(jj(folder, "**", "*.feather"), recursive=True))
        if not strip_path_and_extension(x) in ('document_index', 'token2id')
    ]


def get_document_index_filename(folder: str) -> str:
    """Returns document index filename if exists, otherwise None"""
    return next((x for x in glob.glob(jj(folder, "document_index.feather*"))), None)


def document_index_exists(folder: str | None) -> bool:
    """Returns True if document index exists in folder"""
    return get_document_index_filename(folder) is not None


def read_document_index(folder: str) -> pd.DataFrame:
    """Reads document index from feather file"""

    if filename := get_document_index_filename(folder):
        di: pd.DataFrame = pd.read_feather(filename).set_index('document_name', drop=False)
        sanitize_document_index(di)
        return di

    return None


# def drop_document_index(folder: str) -> None:
#     filename: str = get_document_index_filename(folder)
#     if filename is not None:
#         os.remove(filename)


def write_document_index(folder: str, document_index: pd.DataFrame):
    if document_index is None:
        return

    sanitize_document_index(document_index)
    document_index.reset_index(drop=True).to_feather(jj(folder, 'document_index.feathering'), compression="lz4")


def sanitize_document_index(document_index: pd.DataFrame):
    if document_index is None:
        return

    if '' in document_index.columns:
        document_index.drop(columns='', inplace=True)

    if document_index.index.name in document_index.columns:
        document_index.rename_axis('', inplace=True)


def is_complete(folder: str, di: pd.DataFrame = None) -> bool:
    """Returns True if all documents in folder are indexed in document index"""
    if di is None:
        di: pd.DataFrame = read_document_index(folder)

    if di is None:
        return False

    on_disk: set[str] = set(strip_path_and_extension(get_document_filenames(folder=folder)))
    in_memory: set[str] = set(strip_path_and_extension(di.document_name))
    return on_disk == in_memory


def write_payload(folder: str, payload: DocumentPayload) -> DocumentPayload:
    filename: str = jj(folder, replace_extension(payload.filename, ".feather"))

    payload.content.reset_index(drop=True).to_feather(filename, compression="lz4")

    return payload


def payload_exists(folder: str, payload: DocumentPayload) -> DocumentPayload:
    filename = jj(folder, replace_extension(payload.filename, ".feather"))
    return isfile(filename)


def read_payload(filename: str) -> DocumentPayload:
    filename = replace_extension(filename, ".feather")
    return DocumentPayload(
        content_type=ContentType.TAGGED_FRAME,
        content=pd.read_feather(filename),
        filename=replace_extension(strip_paths(filename), ".csv"),
    )
