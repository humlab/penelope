import fnmatch
import os
import string
import time
from typing import Callable, Iterable, List, Union

from .utils import now_timestamp

VALID_CHARS = "-_.() " + string.ascii_letters + string.digits


def filename_satisfied_by(
    filename: Iterable[str], filename_filter: Union[List[str], Callable], filename_pattern: str = None
) -> bool:

    if filename_pattern is not None:
        if not fnmatch.fnmatch(filename, filename_pattern):
            return False

    if filename_filter is not None:
        if isinstance(filename_filter, list):
            if filename not in filename_filter:
                return False
        elif callable(filename_filter):
            if not filename_filter(filename):
                return False

    return True


def filename_whitelist(filename: str) -> str:
    """Removes invalid characters from filename"""
    filename = ''.join(x for x in filename if x in VALID_CHARS)
    return filename


def path_add_suffix(path: str, suffix: str, new_extension: str = None) -> str:
    basename, extension = os.path.splitext(path)
    return f'{basename}{suffix}{extension if new_extension is None else new_extension}'


def path_add_timestamp(path: str, fmt: str = "%Y%m%d%H%M") -> str:
    return path_add_suffix(path, f'_{time.strftime(fmt)}')


def path_add_date(path: str, fmt: str = "%Y%m%d") -> str:
    return path_add_suffix(path, f'_{time.strftime(fmt)}')


def ts_data_path(directory: str, filename: str):
    return os.path.join(directory, f'{time.strftime("%Y%m%d%H%M")}_{filename}')


def data_path_ts(directory: str, path: str):
    name, extension = os.path.splitext(path)
    return os.path.join(directory, '{}_{}{}'.format(name, time.strftime("%Y%m%d%H%M"), extension))


def path_add_sequence(path: str, i: int, j: int = 0) -> str:
    return path_add_suffix(path, f"_{str(i).zfill(j)}")


def strip_path_and_add_counter(filename: str, i: int, n_zfill: int = 3):
    return f'{os.path.basename(filename)}_{str(i).zfill(n_zfill)}.txt'


def strip_paths(filenames: Union[str, List[str]]) -> Union[str, List[str]]:
    if isinstance(filenames, str):
        return os.path.basename(filenames)
    return [os.path.basename(filename) for filename in filenames]


def strip_path_and_extension(filename: str) -> bool:

    return os.path.splitext(os.path.basename(filename))[0]


def suffix_filename(filename: str, suffix: str) -> str:
    output_path, output_file = os.path.split(filename)
    output_base, output_ext = os.path.splitext(output_file)
    suffixed_filename = os.path.join(output_path, f"{output_base}_{suffix}{output_ext}")
    return suffixed_filename


def replace_extension(filename: str, extension: str) -> str:
    if filename.endswith(extension):
        return filename
    base, _ = os.path.splitext(filename)
    return f"{base}{'' if extension.startswith('.') else '.'}{extension}"


def timestamp_filename(filename: str) -> str:
    return suffix_filename(filename, now_timestamp())
