import fnmatch
import string
import time
from os.path import basename, isdir, join, split, splitext
from typing import Callable, Union

from .utils import now_timestamp

VALID_CHARS = "-_.() " + string.ascii_letters + string.digits


def path_of(path: str):
    return split(path)[0]


def assert_that_path_exists(path: str):
    if path and not isdir(path):
        raise FileNotFoundError(f"folder {path} does not exist")


def filename_satisfied_by(
    filename: str, filename_filter: Union[list[str], Callable], filename_pattern: str = None
) -> bool:
    """Returns true if filename is satisfied by filename filter, and matches pattern"""

    if filename_pattern is not None:
        if not fnmatch.fnmatch(filename, filename_pattern):
            return False

    if filename_filter is None:
        return True

    if isinstance(
        filename_filter,
        (list, set, tuple, dict),
    ):
        # Try both with and without extension
        if filename not in filename_filter and strip_path_and_extension(filename) not in filename_filter:
            return False

    elif callable(filename_filter):
        if not filename_filter(filename):
            return False

    return True


def filenames_satisfied_by(
    filenames: list[str], filename_filter: Union[list[str], Callable] = None, filename_pattern: str = None, sort=True
) -> list[str]:
    """Filters list of filenames based on `filename_pattern` and `filename_filter`

    Args:
        filenames (list[str]): Filenames to filter
        filename_filter (Union[list[str], Callable]): Predicate or list of filenames
        filename_pattern (str, optional): Glob pattern. Defaults to None.

    Returns:
        list[str]: [description]
    """

    satisfied_filenames: list[str] = (
        filenames
        if filename_filter is None and filename_pattern is None
        else [
            filename
            for filename in filenames
            if filename_satisfied_by(filename, filename_filter)
            and (filename_pattern is None or fnmatch.fnmatch(filename, filename_pattern))
        ]
    )

    if sort:
        satisfied_filenames = sorted(satisfied_filenames)

    return satisfied_filenames


def filename_whitelist(filename: str) -> str:
    """Removes invalid characters from filename"""
    filename = ''.join(x for x in filename if x in VALID_CHARS)
    return filename


def split_parts(path: str) -> tuple[str, str, str]:
    """Splits path into folder, filename and extension"""
    folder, filename = split(path)
    base, extension = splitext(filename)
    return folder, base, extension


def path_add_suffix(path: str, suffix: str, new_extension: str = None) -> str:
    base, ext = splitext(path)
    return f'{base}{suffix}{ext if new_extension is None else new_extension}'


def path_add_timestamp(path: str, fmt: str = "%Y%m%d%H%M") -> str:
    return path_add_suffix(path, f'_{time.strftime(fmt)}')


def path_add_date(path: str, fmt: str = "%Y%m%d") -> str:
    return path_add_suffix(path, f'_{time.strftime(fmt)}')


def ts_data_path(directory: str, filename: str):
    return join(directory, f'{time.strftime("%Y%m%d%H%M")}_{filename}')


def data_path_ts(directory: str, path: str):
    name, extension = splitext(path)
    return join(directory, '{}_{}{}'.format(name, time.strftime("%Y%m%d%H%M"), extension))


def path_add_sequence(path: str, i: int, j: int = 0) -> str:
    return path_add_suffix(path, f"_{str(i).zfill(j)}")


def strip_path_and_add_counter(filename: str, i: int, n_zfill: int = 3):
    return f'{basename(strip_extensions(filename))}_{str(i).zfill(n_zfill)}.txt'


def strip_paths(filenames: Union[str, list[str]]) -> Union[str, list[str]]:
    if isinstance(filenames, str):
        return basename(filenames)
    return [basename(filename) for filename in filenames]


def strip_path_and_extension(filename: str | list[str]) -> str | list[str]:
    return strip_extensions(strip_paths(filename))


def strip_extensions(filename: Union[str, list[str]]) -> list[str]:
    if isinstance(filename, str):
        return splitext(filename)[0]
    return [splitext(x)[0] for x in filename]


def suffix_filename(filename: str, suffix: str) -> str:
    folder, base, extension = split_parts(filename)
    return join(folder, f"{base}_{suffix}{extension}")


def replace_extension(filename: str | list[str], extension: str) -> str:
    extension = '' if extension is None else '.' + extension if not extension.startswith('.') else extension
    if isinstance(filename, list):
        return [f"{splitext(f)[0]}{extension}" for f in filename]
    return f"{splitext(filename)[0]}{extension}"


def replace_folder(filename: str | list[str], folder: str) -> str:
    """Replaces folder in filename"""
    if folder is None:
        folder = ''
    if isinstance(filename, list):
        return [join(folder, basename(name)) for name in filename]
    return join(folder or '', basename(filename))


def replace_folder_and_extension(filename: str | list[str], folder: str, extension: str) -> str:
    return replace_extension(replace_folder(filename, folder), extension)


def timestamp_filename(filename: str) -> str:
    return suffix_filename(filename, now_timestamp())


def filter_names_by_pattern(filenames: list[str], filename_pattern: str) -> list[str]:
    return [
        filename for filename in filenames if (filename_pattern is None or fnmatch.fnmatch(filename, filename_pattern))
    ]
