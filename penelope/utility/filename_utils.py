import fnmatch
import os
import re
import string
import sys
import time
from typing import Callable, Dict, Iterable, List, Sequence, Union

from .utils import now_timestamp

IndexOfSplitOrCallableOrRegExp = Union[List[str], Dict[str, Union[Callable, str]]]
ExtractedFilenameFields = Dict[str, Union[int, str]]
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


def filename_field_indexed_split_parser(filename_fields: List[str]):
    """Parses a list of meta-field expressions into a format (kwargs) suitable for `extract_filename_fields`
    The meta-field expressions must either of:
        `fieldname:regexp`
        `fieldname:sep:position`

    Parameters
    ----------
    meta_fields : [type]
        [description]
    """

    def extract_field(data):

        if len(data) == 1:  # regexp
            return data[0]

        if len(data) == 2:  #
            sep = data[0]
            position = int(data[1])
            return lambda f: f.replace('.', sep).split(sep)[position]

        raise ValueError("to many parts in extract expression")

    try:

        filename_fields = {x[0]: extract_field(x[1:]) for x in [y.split(':') for y in filename_fields]}

        return filename_fields

    except:  # pylint: disable=bare-except
        print("parse error: meta-fields, must be in format 'name:regexp'")
        sys.exit(-1)


def extract_filename_fields(filename: str, filename_fields: IndexOfSplitOrCallableOrRegExp) -> ExtractedFilenameFields:
    """Extracts metadata from filename

    The extractor in kwargs must be either a regular expression that extracts the single value
    or a callable function that given the filename return corresponding value.

    Parameters
    ----------
    filename : str
        Filename (basename)
    kwargs: Dict[str, Union[Callable, str]]
        key=extractor list

    Returns
    -------
    Dict[str,Union[int,str]]
        Each key in kwargs is extacted and stored in the dict.

    """

    def astype_int_or_str(v):

        return int(v) if v is not None and v.isnumeric() else v

    def regexp_extract(compiled_regexp, filename: str) -> str:
        try:
            return compiled_regexp.match(filename).groups()[0]
        except:  # pylint: disable=bare-except
            return None

    def fxify(fx_or_re) -> Callable:

        if callable(fx_or_re):
            return fx_or_re

        try:
            compiled_regexp = re.compile(fx_or_re)
            return lambda filename: regexp_extract(compiled_regexp, filename)
        except re.error:
            pass

        return lambda x: fx_or_re  # Return constant expression

    basename = os.path.basename(filename)

    if filename_fields is None:
        return {}

    if isinstance(filename_fields, (list, tuple)):
        # List of `key:sep:index`
        filename_fields = filename_field_indexed_split_parser(filename_fields)

    if isinstance(filename_fields, str):
        # List of `key:sep:index`
        filename_fields = filename_field_indexed_split_parser(filename_fields.split('#'))

    key_fx = {key: fxify(fx_or_re) for key, fx_or_re in filename_fields.items()}

    data = {'filename': basename}
    for key, fx in key_fx.items():
        data[key] = astype_int_or_str(fx(basename))

    return data


def extract_filenames_fields(
    *, filenames: str, filename_fields: Sequence[IndexOfSplitOrCallableOrRegExp]
) -> List[ExtractedFilenameFields]:
    return [
        {'filename': filename, **extract_filename_fields(filename, filename_fields)}
        for filename in strip_paths(filenames)
    ]


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
    suffix = str(i).zfill(j)
    return path_add_suffix(path, suffix)


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
