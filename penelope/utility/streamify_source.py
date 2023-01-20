import fnmatch
import glob
import re
import types
import zipfile
from multiprocessing import Pool
from os.path import isdir, isfile, join
from typing import Any, AnyStr, Callable, Iterable, Iterator, Optional, Union

from . import zip_utils
from .file_utility import read_textfile, read_textfile2
from .filename_utils import filename_satisfied_by, replace_paths


def _read_file_in_zip(args: tuple) -> tuple[str, AnyStr]:
    return zip_utils.read_file_content2(zip_or_filename=args[0], filename=args[1], as_binary=args[2])


def _read_file_in_folder(args: tuple) -> tuple[str, AnyStr]:
    return read_textfile2(filename=args[0], as_binary=args[1])


def streamify_zip_source(
    *,
    path: zipfile.ZipFile,
    filenames: list[str] = None,
    filename_pattern='*.*',
    filename_filter: Union[list[str], Callable] = None,
    as_binary: bool = False,
    n_processes: int = 1,
    n_chunksize: int = 5,
) -> Iterator[tuple[str, AnyStr]]:

    filenames = filenames or zip_utils.list_filenames(
        zip_or_filename=path, filename_pattern=filename_pattern, filename_filter=filename_filter
    )

    if n_processes == 1:

        with zipfile.ZipFile(path, 'r') as zf:

            for filename in filenames:
                yield zip_utils.read_file_content2(zip_or_filename=zf, filename=filename, as_binary=as_binary)

    else:

        args: str = [(path, filename, as_binary) for filename in filenames]

        with Pool(processes=n_processes) as pool:
            futures = pool.imap(_read_file_in_zip, args, chunksize=n_chunksize)
            for data in futures:
                yield data


def streamify_folder_source(
    path: str,
    filenames: Optional[list[str]] = None,
    filename_pattern: str = "*.*",
    filename_filter: Union[list[str], Callable] = None,
    as_binary: bool = False,
    n_processes: int = 1,
    n_chunksize: int = 5,
) -> Iterable[tuple[str, AnyStr]]:

    if filenames is None:
        filenames = list_any_source(path, filename_pattern=filename_pattern, filename_filter=filename_filter)

    filenames: list[str] = replace_paths(path, filenames)

    if n_processes == 1:

        for filename in filenames:
            yield read_textfile2(filename, as_binary)

    else:

        with Pool(processes=n_processes) as executor:
            args = [(filename, as_binary) for filename in filenames]
            for data in executor.imap(_read_file_in_folder, args, chunksize=n_chunksize):
                yield data


def streamify_any_source(  # pylint: disable=too-many-return-statements
    source: Union[AnyStr, zipfile.ZipFile, list, Any],
    filenames: list[str] = None,
    filename_pattern: str = '*.*',
    filename_filter: Union[list[str], Callable] = None,
    as_binary: bool = False,
    n_processes: int = 1,
    n_chunksize: int = 5,
) -> Iterable[tuple[str, AnyStr]]:

    filenames = filenames or list_any_source(source, filename_pattern=filename_pattern, filename_filter=filename_filter)

    if not isinstance(source, (str, zipfile.ZipFile)):
        if hasattr(source, '__iter__') and hasattr(source, '__next__'):
            return source

    if isinstance(source, list):

        if len(source) == 0:
            return []

        if isinstance(source[0], tuple):
            return source

        return ((f'document_{i+1}.txt', d) for i, d in enumerate(source))

    if zipfile.is_zipfile(source):

        return streamify_zip_source(
            path=source,
            filenames=filenames,
            filename_pattern=filename_pattern,
            filename_filter=filename_filter,
            as_binary=as_binary,
            n_processes=n_processes,
            n_chunksize=n_chunksize,
        )

    if isinstance(source, str):

        if isdir(source):
            return streamify_folder_source(
                path=source,
                filenames=filenames,
                filename_pattern=filename_pattern,
                filename_filter=filename_filter,
                as_binary=as_binary,
                n_processes=n_processes,
                n_chunksize=n_chunksize,
            )

        if isfile(source):
            return iter([read_textfile2(source, as_binary=as_binary)])

        return iter([('document', source)])

    raise FileNotFoundError(source)


def list_any_source(
    text_source: Union[str, zipfile.ZipFile, list],
    filename_pattern: str = "*.txt",
    filename_filter=None,
) -> list[str]:
    """Returns all filenames that matches `pattern` in archive

    Parameters
    ----------
    folder_or_zip : str
        File pattern

    Returns
    -------
    list[str]
        list of filenames
    """

    filenames = None

    if isinstance(text_source, zipfile.ZipFile):

        filenames = text_source.namelist()

    elif isinstance(text_source, str):

        if isfile(text_source):

            if zipfile.is_zipfile(text_source):

                with zipfile.ZipFile(text_source) as zf:
                    filenames = zf.namelist()

            else:
                filenames = [text_source]

        elif isdir(text_source):

            filenames = glob.glob(join(text_source, filename_pattern))

    elif isinstance(text_source, types.GeneratorType):
        filenames = (x[0] for x in text_source)

    elif isinstance(text_source, list):

        if len(text_source) == 0:
            filenames = []

        if isinstance(text_source[0], tuple):
            filenames = [x[0] for x in text_source]
        else:
            filenames = [f'document_{i+1}.txt' for i in range(0, len(text_source))]

    if filenames is None:

        raise ValueError(f"Source '{text_source}' not found. Only folder or ZIP or file are valid arguments")

    return [
        filename
        for filename in sorted(filenames)
        if filename_satisfied_by(filename, filename_filter)
        and (filename_pattern is None or fnmatch.fnmatch(filename, filename_pattern))
    ]


def _is_zipfile(source: str) -> bool:
    try:
        return isinstance(source, zipfile.ZipFile) or (isinstance(source, str) and isfile(source))
    except:  # pylint: disable=bare-except
        return False


DJANGO_URL_VALIDATOR = re.compile(
    r'^(?:http|ftp)s?://'  # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
    r'localhost|'  # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)$',
    re.IGNORECASE,
)


def is_url(source: str) -> bool:

    return isinstance(source, str) and bool(DJANGO_URL_VALIDATOR.search(source))


def read_text(source: Any, filename: str) -> str:

    if _is_zipfile(source):
        _, data = zip_utils.read_file_content2(zip_or_filename=source, filename=filename, as_binary=False)
        return data

    if isinstance(source, str) and isdir(source) and isfile(join(source, filename)):

        return read_textfile(join(source, filename))

    # if is_url(source):
    #     ...

    raise FileNotFoundError(filename)
