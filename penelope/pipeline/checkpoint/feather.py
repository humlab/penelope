import glob
from os.path import isfile
from os.path import join as jj

import pandas as pd

from penelope.utility import replace_extension, strip_path_and_extension


def to_document_filename(folder: str, document_name: str) -> str:
    """Returns document filename"""
    return jj(folder, replace_extension(document_name, '.feather'))


def get_document_filenames(*, folder: str) -> list[str]:
    """Returns list of document filenames in folder"""
    return [
        x
        for x in sorted(glob.glob(jj(folder, "**", "*.feather"), recursive=True))
        if strip_path_and_extension(x) not in ('document_index', 'token2id')
    ]


def get_document_index_filename(folder: str) -> str:
    """Returns document index filename if exists, otherwise None"""
    if folder is None:
        return None
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


def write_document_index(folder: str, document_index: pd.DataFrame):
    if document_index is None:
        return

    sanitize_document_index(document_index)
    # FIXME: Change filename extension to '.feather' as it should be.
    # Add logic that separates documents and metadata in another way
    document_index.reset_index(drop=True).to_feather(jj(folder, 'document_index.feathering'), compression="lz4")


def sanitize_document_index(document_index: pd.DataFrame):
    if document_index is None:
        return

    if '' in document_index.columns:
        document_index.drop(columns='', inplace=True)

    if document_index.index.name in document_index.columns:
        document_index.rename_axis('', inplace=True)


def write_document(document: pd.DataFrame, filename: str, force: bool = False):
    if force or not isfile(filename):
        document.reset_index(drop=True).to_feather(filename, compression="lz4")


def is_complete(folder: str, di: pd.DataFrame = None) -> bool:
    """Returns True if all documents in folder are indexed in document index"""
    if di is None:
        di: pd.DataFrame = read_document_index(folder)

    if di is None:
        return False

    on_disk: set[str] = set(strip_path_and_extension(get_document_filenames(folder=folder)))
    in_memory: set[str] = set(strip_path_and_extension(di.document_name))
    return on_disk == in_memory
