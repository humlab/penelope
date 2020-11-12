import fnmatch
import glob
import json
import logging
import os
import pathlib
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple, Union

import gensim
import pandas as pd

from .filename_utils import filename_satisfied_by

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


def default_data_folder():
    home = Path.home()
    home_data = os.path.join(str(home), "data")
    if os.path.isdir(home_data):
        return home_data
    if os.path.isdir('/data'):
        return '/data'
    return str(home)


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


def store_to_archive(archive_name: str, stream: Iterable[Tuple[str, Iterable[str]]]):
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


def read_from_archive(folder_or_zip: Union[str, zipfile.ZipFile], filename: str, as_binary=False) -> str:
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


def excel_to_csv(excel_file: str, text_file: str, sep: str = '\t') -> pd.DataFrame:
    """Exports Excel to a tab-seperated text file"""
    df = pd.read_excel(excel_file)
    df.to_csv(text_file, sep=sep)
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


def compress_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    folder, filename = os.path.split(path)
    name, _ = os.path.splitext(filename)
    zip_name = os.path.join(folder, name + '.zip')
    with zipfile.ZipFile(zip_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(path)
    os.remove(path)


def zip_get_filenames(zip_filename: str, extension: str = '.txt') -> List[str]:
    with zipfile.ZipFile(zip_filename, mode='r') as zf:
        return [x for x in zf.namelist() if x.endswith(extension)]


def zip_get_text(zip_filename: str, filename: str) -> str:
    with zipfile.ZipFile(zip_filename, mode='r') as zf:
        return zf.read(filename).decode(encoding='utf-8')


def read_json(path: str) -> Dict:
    """Reads JSON from file"""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path) as fp:
        return json.load(fp)


def write_json(path: str, data: Dict):
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
