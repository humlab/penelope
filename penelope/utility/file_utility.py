import fnmatch
import glob
import logging
import os
import pathlib
import re
import sys
import time
import zipfile
from typing import Callable, Dict, Iterable, Iterator, List, Tuple, Union

import gensim
import pandas as pd

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


def strip_path_and_extension(filename: str) -> bool:

    return os.path.splitext(os.path.basename(filename))[0]


def strip_path_and_add_counter(filename, n_chunk):

    return '{}_{}.txt'.format(os.path.basename(filename), str(n_chunk).zfill(3))


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


# def basename(path: str):
#     return os.path.splitext(os.path.basename(path))[0]


def basenames(filenames: List[str]) -> List[str]:
    return [os.path.basename(filename) for filename in filenames]


# TODO: Merge with penelope.corpus.readers.streamify_text_source?
def create_iterator(
    folder_or_zip: str, filenames: List[str] = None, filename_pattern: str = '*.txt', as_binary: bool = False
) -> Tuple[str, Iterator[str]]:

    filenames = filenames or list_filenames(folder_or_zip, filename_pattern=filename_pattern)

    if not isinstance(folder_or_zip, str):
        raise ValueError("folder_or_zip argument must be a path")

    if os.path.isfile(folder_or_zip):
        with zipfile.ZipFile(folder_or_zip) as zip_file:

            for filename in filenames:

                with zip_file.open(filename, 'r') as text_file:

                    content = text_file.read() if as_binary else text_file.read().decode('utf-8')

                yield os.path.basename(filename), content

    elif os.path.isdir(folder_or_zip):
        for filename in filenames:
            content = read_textfile(filename)
            yield os.path.basename(filename), content
    else:
        raise FileNotFoundError(folder_or_zip)


def list_filenames(
    text_source: Union[str, zipfile.ZipFile, List], filename_pattern: str = "*.txt", filename_filter=None
) -> List[str]:
    """Returns all filenames that matches `pattern` in archive

    Parameters
    ----------
    folder_or_zip : str
        File pattern

    Returns
    -------
    List[str]
        List of filenames
    """

    filenames = None

    if isinstance(text_source, zipfile.ZipFile):

        filenames = text_source.namelist()

    elif isinstance(text_source, str):

        if os.path.isfile(text_source):

            if zipfile.is_zipfile(text_source):

                with zipfile.ZipFile(text_source) as zf:
                    filenames = zf.namelist()

            else:
                filenames = [text_source]

        elif os.path.isdir(text_source):

            filenames = glob.glob(os.path.join(text_source, filename_pattern))

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


def store(archive_name: str, stream: Iterable[Tuple[str, Iterable[str]]]):
    """Stores stream of text [(name, tokens), (name, tokens), ..., (name, tokens)] as text files in a new zip-file

    Parameters
    ----------
    archive_name : str
        Target filename
    stream : List[Tuple[str, Union[List[str], str]]]
        Documents [(name, tokens), (name, tokens), ..., (name, tokens)]
    """
    with zipfile.ZipFile(archive_name, 'w', compresslevel=zipfile.ZIP_DEFLATED) as out:

        for (filename, document) in stream:

            data = document if isinstance(document, str) else ' '.join(document)
            out.writestr(filename, data, compresslevel=zipfile.ZIP_DEFLATED)


def read(folder_or_zip: Union[str, zipfile.ZipFile], filename: str, as_binary=False) -> str:
    """Returns content in file `filename` that exists in folder or zip `folder_or_zip`

    Parameters
    ----------
    folder_or_zip : Union[str, zipfile.ZipFile]
        Folder (if `filename` is file in folder) or ZIP-filename
    filename : str
        Filename in folder or ZIP-file
    as_binary : bool, optional
        Opens file in binary mode, by default False

    Returns
    -------
    str
        File content

    Raises
    ------
    IOError
        If file not found or cannot be read
    """
    if isinstance(folder_or_zip, zipfile.ZipFile):
        with folder_or_zip.open(filename, 'r') as f:
            return f.read() if as_binary else f.read().decode('utf-8')

    if os.path.isdir(folder_or_zip):

        path = os.path.join(folder_or_zip, filename)

        if os.path.isfile(path):
            with open(path, 'r') as f:
                return gensim.utils.to_unicode(f.read(), 'utf8', errors='ignore')

    if os.path.isfile(folder_or_zip):

        if zipfile.is_zipfile(folder_or_zip):

            with zipfile.ZipFile(folder_or_zip) as zf:
                with zf.open(filename, 'r') as f:
                    return f.read() if as_binary else f.read().decode('utf-8')

        else:
            return read_textfile(folder_or_zip)

    raise IOError("File not found")


def read_textfile(filename: str, as_binary: bool = False) -> str:

    opts = {'mode': 'rb'} if as_binary else {'mode': 'r', 'encoding': 'utf-8'}
    with open(filename, **opts) as f:
        try:
            data = f.read()
            content = data  # .decode('utf-8')
        except UnicodeDecodeError:
            print('UnicodeDecodeError: {}'.format(filename))
            # content = data.decode('cp1252')
            raise
        return content


# def read_file(path, filename):
#     if os.path.isdir(path):
#         with open(os.path.join(path, filename), 'r') as file:
#             content = file.read()
#     else:
#         with zipfile.ZipFile(path) as zf:
#             with zf.open(filename, 'r') as file:
#                 content = file.read()
#     content = gensim.utils.to_unicode(content, 'utf8', errors='ignore')
#     return content


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


IndexOfSplitOrCallableOrRegExp = Union[List[str], Dict[str, Union[Callable, str]]]
FilenameFields = Dict[str, Union[int, str]]


def extract_filename_fields(filename: str, filename_fields: IndexOfSplitOrCallableOrRegExp) -> FilenameFields:
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


# def extract_filename_fields_list(filenames: List[str], filename_fields: IndexOfSplitOrCallableOrRegExp) -> List[FilenameFields]:

#     return [extract_filename_fields(x, filename_fields) for x in filenames]


def export_excel_to_text(excel_file: str, text_file: str) -> pd.DataFrame:
    """Exports Excel to a tab-seperated text file"""
    df = pd.read_excel(excel_file)
    df.to_csv(text_file, sep='\t')
    return df


def read_text_file(filename: str) -> pd.DataFrame:
    """Exports Excel to a tab-seperated text file"""
    df = pd.read_csv(filename, sep='\t')  # [['year', 'txt']]
    return df


def find_parent_folder(name: str) -> str:
    path = pathlib.Path(os.getcwd())
    folder = os.path.join(*path.parts[: path.parts.index(name) + 1])
    return folder


def find_parent_folder_with_child(folder: str, target: str) -> pathlib.Path:
    path = pathlib.Path(folder).resolve()
    while path is not None:
        name = os.path.join(path, target)
        if os.path.isfile(name) or os.path.isdir(name):
            return path
        if path in ('', '/'):
            break
        path = path.parent
    return None


def find_folder(folder: str, parent: str) -> str:
    return os.path.join(folder.split(parent)[0], parent)


def read_excel(filename: str, sheet: str) -> pd.DataFrame:
    if not os.path.isfile(filename):
        raise Exception("File {0} does not exist!".format(filename))
    with pd.ExcelFile(filename) as xls:
        return pd.read_excel(xls, sheet)


def save_excel(data: pd.DataFrame, filename: str):
    with pd.ExcelWriter(filename) as writer:  # pylint: disable=abstract-class-instantiated
        for (df, name) in data:
            df.to_excel(writer, name, engine='xlsxwriter')
        writer.save()


def ts_data_path(directory: str, filename: str):
    return os.path.join(directory, '{}_{}'.format(time.strftime("%Y%m%d%H%M"), filename))


def data_path_ts(directory: str, path: str):
    name, extension = os.path.splitext(path)
    return os.path.join(directory, '{}_{}{}'.format(name, time.strftime("%Y%m%d%H%M"), extension))


def compress_file(path: str):
    if not os.path.exists(path):
        # logger.error("ERROR: file not found (zip)")
        return
    folder, filename = os.path.split(path)
    name, _ = os.path.splitext(filename)
    zip_name = os.path.join(folder, name + '.zip')
    with zipfile.ZipFile(zip_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(path)
    os.remove(path)
