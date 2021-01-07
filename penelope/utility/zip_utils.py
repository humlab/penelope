import csv
import json
import os
import zipfile
from fnmatch import fnmatch
from functools import wraps
from io import StringIO
from typing import Iterable, List, Tuple, Union

import pandas as pd
from penelope.utility.filename_utils import replace_extension, strip_path_and_extension

ZipFileOrStr = Union[str, zipfile.ZipFile]


def zipfile_or_filename(**zipargs):
    def zipfile_or_str_outer(func):
        @wraps(func)
        def zipfile_or_str_inner(zip_or_filename: ZipFileOrStr, **kwargs):
            if isinstance(zip_or_filename, zipfile.ZipFile):
                return func(zip_or_filename=zip_or_filename, **kwargs)
            with zipfile.ZipFile(zip_or_filename, **zipargs) as zf:
                return func(zip_or_filename=zf, **kwargs)

        return zipfile_or_str_inner

    return zipfile_or_str_outer


@zipfile_or_filename(mode='r')
def namelist(*, zip_or_filename: zipfile.ZipFile, pattern: str = '*.txt') -> List[str]:
    return [x for x in zip_or_filename.namelist() if fnmatch(x, pattern)]


@zipfile_or_filename(mode='r')
def read(*, zip_or_filename: zipfile.ZipFile, filename: str, as_binary=False) -> str:
    return zip_or_filename.read(filename) if as_binary else zip_or_filename.read(filename).decode(encoding='utf-8')


def read_iterator(*, path: zipfile.ZipFile, filenames: List[str] = None, pattern='*.*', as_binary: bool = False):
    with zipfile.ZipFile(path, 'r') as zf:
        filenames = filenames or namelist(zip_or_filename=zf, pattern=pattern)
        for filename in filenames:
            with zf.open(filename, 'r') as fp:
                content = fp.read() if as_binary else fp.read().decode('utf-8')
            yield os.path.basename(filename), content


@zipfile_or_filename(mode='w', compresslevel=zipfile.ZIP_DEFLATED)
def store(*, zip_or_filename: zipfile.ZipFile, stream: Iterable[Tuple[str, Union[str, Iterable[str]]]]):
    """Stores token stream to archive
    Args:
        zf (zipfile.ZipFile): [description]
        stream (Iterable[Tuple[str, Iterable[str]]]): [description]
    """
    for (filename, document) in stream:
        data: str = document if isinstance(document, str) else ' '.join(document)
        zip_or_filename.writestr(filename, data, compresslevel=zipfile.ZIP_DEFLATED)


def compress(path: str, remove: bool = True):
    """Compresses a file on disk"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    filename = replace_extension(path, '.zip')

    with zipfile.ZipFile(filename, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(path)

    if remove:
        os.remove(path)


def unpack(path: str, target_folder: str, create_sub_folder: bool = True):
    """Unpacks zip to specified folder"""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    if not os.path.isdir(target_folder):
        raise FileNotFoundError(target_folder)

    if create_sub_folder:
        target_folder = os.path.join(target_folder, strip_path_and_extension(path))
        os.makedirs(target_folder, exist_ok=True)

    with zipfile.ZipFile(path, "r") as z:
        z.extractall(target_folder)


@zipfile_or_filename(mode='r')
def read_json(*, zip_or_filename: zipfile.ZipFile, filename: str, as_binary: bool = False) -> dict:
    return json.loads(read(zip_or_filename=zip_or_filename, filename=filename, as_binary=as_binary))


@zipfile_or_filename(mode='r')
def read_dataframe(
    *, zip_or_filename: zipfile.ZipFile, filename: str, sep: str = '\t', quoting: int = csv.QUOTE_NONE
) -> pd.DataFrame:
    data_str = read(zip_or_filename=zip_or_filename, filename=filename, as_binary=False)
    df = pd.read_csv(StringIO(data_str), sep=sep, quoting=quoting, index_col=0)
    return df
