import os
import zipfile
from fnmatch import fnmatch
from functools import wraps
from typing import Iterable, List, Tuple, Union

from penelope.utility.filename_utils import replace_extension, strip_path_and_extension

ZipFileOrStr = Union[str, zipfile.ZipFile]


def zipfile_or_str(**zipargs):
    def zipfile_or_str_outer(func):
        @wraps(func)
        def zipfile_or_str_inner(zip_or_str: ZipFileOrStr, **kwargs):
            if isinstance(zip_or_str, zipfile.ZipFile):
                return func(zip_or_str=zip_or_str, **kwargs)
            with zipfile.ZipFile(zip_or_str, **zipargs) as zf:
                return func(zip_or_str=zf, **kwargs)
        return zipfile_or_str_inner
    return zipfile_or_str_outer


@zipfile_or_str(mode='r')
def namelist(*, zip_or_str: zipfile.ZipFile, pattern: str = '*.txt') -> List[str]:
    return [x for x in zip_or_str.namelist() if fnmatch(x, pattern)]


@zipfile_or_str(mode='r')
def read(*, zip_or_str: zipfile.ZipFile, filename: str, as_binary=False) -> str:
    return zip_or_str.read(filename) if as_binary else zip_or_str.read(filename).decode(encoding='utf-8')


def read_iterator(*, path: zipfile.ZipFile, filenames: List[str] = None, pattern='*.*', as_binary: bool = False):
    with zipfile.ZipFile(path, 'r') as zf:
        filenames = filenames or namelist(zip_or_str=zf, pattern=pattern)
        for filename in filenames:
            with zf.open(filename, 'r') as fp:
                content = fp.read() if as_binary else fp.read().decode('utf-8')
            yield os.path.basename(filename), content


@zipfile_or_str(mode='w', compresslevel=zipfile.ZIP_DEFLATED)
def store(*, zip_or_str: zipfile.ZipFile, stream: Iterable[Tuple[str, Union[str, Iterable[str]]]]):
    """Stores token stream to archive
    Args:
        zf (zipfile.ZipFile): [description]
        stream (Iterable[Tuple[str, Iterable[str]]]): [description]
    """
    for (filename, document) in stream:
        data: str = document if isinstance(document, str) else ' '.join(document)
        zip_or_str.writestr(filename, data, compresslevel=zipfile.ZIP_DEFLATED)


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
