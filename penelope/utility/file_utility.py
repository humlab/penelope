import bz2
import fnmatch
import glob
import json
import logging
import os
import pathlib
import pickle
import zipfile
from io import StringIO
from os.path import basename, exists, isdir, isfile, join
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import pandas as pd

from . import filename_utils as utils
from . import zip_utils
from .filename_utils import filename_satisfied_by, replace_paths

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


def default_data_folder():
    home = Path.home()
    home_data = join(str(home), "data")
    if isdir(home_data):
        return home_data
    if isdir('/data'):
        return '/data'
    return str(home)


def folder_read_iterator(folder: str, filenames: List[str]) -> Iterable[Tuple[str, str]]:
    for filename in replace_paths(folder, filenames):
        data = read_textfile(filename)
        yield basename(filename), data


# TODO: Merge with penelope.corpus.readers.streamify_text_source?
def create_iterator(
    folder_or_zip: str, filenames: List[str] = None, filename_pattern: str = '*.txt', as_binary: bool = False
) -> Iterable[Tuple[str, str]]:

    filenames = filenames or list_filenames(folder_or_zip, filename_pattern=filename_pattern)

    if not isinstance(folder_or_zip, str):
        raise ValueError("folder_or_zip argument must be a path")

    if isfile(folder_or_zip):
        return zip_utils.read_iterator(path=folder_or_zip, filenames=filenames, as_binary=as_binary)

    if isdir(folder_or_zip):
        return folder_read_iterator(folder=folder_or_zip, filenames=filenames)

    raise FileNotFoundError(folder_or_zip)


# def read_from_archive(folder_or_zip: Union[str, zipfile.ZipFile], filename: str, as_binary=False) -> str:

#     if isinstance(folder_or_zip, zipfile.ZipFile):
#         with folder_or_zip.open(filename, 'r') as f:
#             return f.read() if as_binary else f.read().decode('utf-8')

#     if isdir(folder_or_zip):

#         path = join(folder_or_zip, filename)

#         if isfile(path):
#             with open(path, 'r') as f:
#                 return gensim.utils.to_unicode(f.read(), 'utf8', errors='ignore')

#     if isfile(folder_or_zip):

#         if zipfile.is_zipfile(folder_or_zip):
#             return zip_utils.read(zip_or_name=folder_or_zip, filename=filename, as_binary=as_binary)

#         return read_textfile(folder_or_zip)

#     raise IOError("File not found")


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

        if isfile(text_source):

            if zipfile.is_zipfile(text_source):

                with zipfile.ZipFile(text_source) as zf:
                    filenames = zf.namelist()

            else:
                filenames = [text_source]

        elif isdir(text_source):

            filenames = glob.glob(join(text_source, filename_pattern))

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
    folder = join(*path.parts[: path.parts.index(name) + 1])
    return folder


def find_parent_folder_with_child(folder: str, target: str) -> pathlib.Path:
    path = pathlib.Path(folder).resolve()
    while path is not None:
        name = join(path, target)
        if isfile(name) or isdir(name):
            return path
        if path in ('', '/'):
            break
        path = path.parent
    return None


def find_folder(folder: str, parent: str) -> str:
    return join(folder.split(parent)[0], parent)


def read_excel(filename: str, sheet: str) -> pd.DataFrame:
    if not isfile(filename):
        raise Exception("File {0} does not exist!".format(filename))
    with pd.ExcelFile(filename) as xls:
        return pd.read_excel(xls, sheet)


def save_excel(data: pd.DataFrame, filename: str):
    with pd.ExcelWriter(filename) as writer:  # pylint: disable=abstract-class-instantiated
        for (df, name) in data:
            df.to_excel(writer, name, engine='xlsxwriter')
        writer.save()


def read_json(path: str) -> Dict:
    """Reads JSON from file"""
    if not isfile(path):
        raise FileNotFoundError(path)
    with open(path) as fp:
        return json.load(fp)


def write_json(path: str, data: Dict, default=None):
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4, default=default)


DataFrameFilenameTuple = Tuple[pd.DataFrame, str]


def pandas_to_csv_zip(
    zip_filename: str, dfs: Union[DataFrameFilenameTuple, List[DataFrameFilenameTuple]], extension='csv', **to_csv_opts
):
    if not isinstance(dfs, (list, tuple)):
        raise ValueError("expected tuple or list of tuples")

    if isinstance(dfs, (tuple,)):
        dfs = [dfs]

    with zipfile.ZipFile(zip_filename, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for (df, filename) in dfs:
            if not isinstance(df, pd.core.frame.DataFrame) or not isinstance(filename, str):
                raise ValueError(
                    f"Expected Tuple[pd.DateFrame, filename: str], found Tuple[{type(df)}, {type(filename)}]"
                )
            filename = utils.replace_extension(filename=filename, extension=extension)
            data_str = df.to_csv(**to_csv_opts)
            zf.writestr(filename, data=data_str)


def pandas_read_csv_zip(zip_filename: str, pattern='*.csv', **read_csv_opts) -> Dict:

    data = dict()
    with zipfile.ZipFile(zip_filename, mode='r') as zf:
        for filename in zf.namelist():
            if not fnmatch.fnmatch(filename, pattern):
                logging.info(f"skipping {filename} down't match {pattern} ")
                continue
            df = pd.read_csv(StringIO(zf.read(filename).decode(encoding='utf-8')), **read_csv_opts)
            data[filename] = df
    return data


def pickle_compressed_to_file(filename: str, thing: Any):
    with bz2.BZ2File(filename, 'w') as f:
        pickle.dump(thing, f)


def unpickle_compressed_from_file(filename: str):
    with bz2.BZ2File(filename, 'rb') as f:
        data = pickle.load(f)
        return data


def pickle_to_file(filename: str, thing: Any):
    """Pickles a thing to disk """
    if filename.endswith('.pbz2'):
        pickle_compressed_to_file(filename, thing)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(thing, f, pickle.HIGHEST_PROTOCOL)


def unpickle_from_file(filename: str) -> Any:
    """Unpickles a thing from disk."""
    if filename.endswith('.pbz2'):
        thing = unpickle_compressed_from_file(filename)
    else:
        with open(filename, 'rb') as f:
            thing = pickle.load(f)
    return thing


def symlink_files(source_pattern: str, target_folder: str) -> None:
    os.makedirs(target_folder, exist_ok=True)
    for f in glob.glob(source_pattern):
        t = join(target_folder, basename(f))
        if not exists(t):
            os.symlink(f, t)
