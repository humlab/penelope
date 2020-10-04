import fnmatch
import glob
import logging
import os
import pathlib
import re
import sys
import time
import types
import zipfile
from typing import Callable, Iterable, List, Tuple, Union

import gensim
import pandas as pd

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


def strip_path_and_extension(filename):

    return os.path.splitext(os.path.basename(filename))[0]


def strip_path_and_add_counter(filename, n_chunk):

    return '{}_{}.txt'.format(os.path.basename(filename), str(n_chunk).zfill(3))


def filename_satisfied_by(filename: Iterable[str], filename_filter: Union[List[str], Callable]):

    if filename_filter is None:
        return True

    if isinstance(filename_filter, list):
        return filename in filename_filter

    if callable(filename_filter):
        return filename_filter(filename)

    return True


def basename(path):
    return os.path.splitext(os.path.basename(path))[0]


def create_iterator(
    folder_or_zip: str, filenames: List[str] = None, filename_pattern: str = '*.txt', as_binary: bool = False
):

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


def list_filenames(folder_or_zip: Union[str, zipfile.ZipFile], filename_pattern: str = "*.txt", filename_filter=None):
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

    if isinstance(folder_or_zip, zipfile.ZipFile):

        filenames = folder_or_zip.namelist()

    elif isinstance(folder_or_zip, str):

        if os.path.isfile(folder_or_zip):

            if zipfile.is_zipfile(folder_or_zip):

                with zipfile.ZipFile(folder_or_zip) as zf:
                    filenames = zf.namelist()

            else:
                filenames = [folder_or_zip]

        elif os.path.isdir(folder_or_zip):

            filenames = glob.glob(os.path.join(folder_or_zip, filename_pattern))

    if filenames is None:

        raise ValueError(f"Source '{folder_or_zip}' not found. Only folder or ZIP or file are valid arguments")

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


def read(folder_or_zip: Union[str, zipfile.ZipFile], filename: str, as_binary=False):
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


def read_textfile(filename, as_binary=False):

    opts = {'mode': 'rb'} if as_binary else {'mode': 'r', 'encoding': 'utf-8'}
    with open(filename, **opts) as f:
        try:
            data = f.read()
            content = data  # .decode('utf-8')
        except UnicodeDecodeError as _:
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


def filename_field_parser(meta_fields):
    def extract_field(data):

        if len(data) == 1:  # regexp
            return data[0]

        if len(data) == 2:  #
            sep = data[0]
            position = int(data[1])
            return lambda f: f.replace('.', sep).split(sep)[position]

        raise ValueError("to many parts in extract expression")

    try:

        filename_fields = {x[0]: extract_field(x[1:]) for x in [y.split(':') for y in meta_fields]}

        return filename_fields

    except:  # pylint: disable=bare-except
        print("parse error: meta-fields, must be in format 'name:regexp'")
        sys.exit(-1)


def extract_filename_fields(filename, **kwargs):
    """Extracts metadata from filename

    Parameters
    ----------
    filename : str
        Filename (basename)
    kwargs: key=extractor list

    Returns
    -------
    SimpleNamespace
        Each key in kwargs is set as a property in the returned instance.
        The extractor must be either a regular expression that extracts the single value
        or a callable function that given the filename return corresponding value.

    """
    kwargs = kwargs or {}
    params = {x: None for x in kwargs}
    data = types.SimpleNamespace(filename=filename, **params)
    for k, r in kwargs.items():

        if r is None:
            continue

        if callable(r):
            v = r(filename)
            data.__setattr__(k, int(v) if v.isnumeric() else v)

        if isinstance(r, str):  # typing.re.Pattern):
            m = re.match(r, filename)
            if m is not None:
                v = m.groups()[0]
                data.__setattr__(k, int(v) if v.isnumeric() else v)

    return data


def find_parent_folder(name):
    path = pathlib.Path(os.getcwd())
    folder = os.path.join(*path.parts[: path.parts.index(name) + 1])
    return folder


def read_excel(filename, sheet):
    if not os.path.isfile(filename):
        raise Exception("File {0} does not exist!".format(filename))
    with pd.ExcelFile(filename) as xls:
        return pd.read_excel(xls, sheet)


def save_excel(data, filename):
    with pd.ExcelWriter(filename) as writer:  # pylint: disable=abstract-class-instantiated
        for (df, name) in data:
            df.to_excel(writer, name, engine='xlsxwriter')
        writer.save()


def ts_data_path(directory, filename):
    return os.path.join(directory, '{}_{}'.format(time.strftime("%Y%m%d%H%M"), filename))


def data_path_ts(directory, path):
    name, extension = os.path.splitext(path)
    return os.path.join(directory, '{}_{}{}'.format(name, time.strftime("%Y%m%d%H%M"), extension))


def compress_file(path):
    if not os.path.exists(path):
        # logger.error("ERROR: file not found (zip)")
        return
    folder, filename = os.path.split(path)
    name, _ = os.path.splitext(filename)
    zip_name = os.path.join(folder, name + '.zip')
    with zipfile.ZipFile(zip_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(path)
    os.remove(path)
