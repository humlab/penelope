import bz2
import glob
import json
import logging
import os
import pathlib
import pickle
from os.path import basename, exists, isdir, isfile, join
from pathlib import Path
from typing import Any, AnyStr, Dict, List, Mapping, Tuple

import pandas as pd
import yaml

from .filename_utils import replace_extension

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


def default_data_folder():
    home = Path.home()
    home_data = join(str(home), "data")
    if isdir(home_data):
        return home_data
    if isdir('/data'):
        return '/data'
    return str(home)


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


def touch(filename: str) -> str:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    Path(filename).touch()
    return filename


def probe_extension(filename: str, extensions: str = 'feather,csv,zip') -> str:
    """Checks if `filename` exists, or with any of given extensions"""
    if os.path.isfile(filename):
        return filename

    for extension in extensions.split(','):
        probe_name: str = replace_extension(filename, extension.strip())
        if os.path.isfile(probe_name):
            return probe_name

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


def read_json(path: str, default: dict = None) -> Dict:
    """Reads JSON from file"""

    if not isfile(path):

        if default:
            return default

        raise FileNotFoundError(path)

    with open(path) as fp:
        return json.load(fp)


def write_json(path: str, data: Dict, default=None):

    if default is not None:
        if not callable(default):
            default = lambda _: default

    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4, default=default)


def pickle_compressed_to_file(filename: str, thing: Any):
    with bz2.BZ2File(filename, 'w') as f:
        pickle.dump(thing, f)


def unpickle_compressed_from_file(filename: str):
    with bz2.BZ2File(filename, 'rb') as f:
        data = pickle.load(f)
        return data


def pickle_to_file(filename: str, thing: Any):
    """Pickles a thing to disk"""
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


def read_textfile(filename: str, as_binary: bool = False) -> str:
    """Returns text content from `filename`"""
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


def read_textfile2(filename: str, as_binary: bool = False) -> Tuple[str, AnyStr]:
    """Reads text in `filename` and return a tuple filename and text"""
    data = read_textfile(filename, as_binary=as_binary)
    return basename(filename), data


def load_term_substitutions(
    filepath: str, default_term: str = '_masked_', delim: str = ';', vocab=None
) -> Mapping[str, str]:
    """Load term substitution map from file. Return dict."""
    substitutions: Mapping[str, str] = {}

    if not os.path.isfile(filepath):
        return {}

    with open(filepath) as f:
        substitutions = {
            x[0].strip(): x[1].strip()
            for x in (tuple(line.lower().split(delim)) + (default_term,) for line in f.readlines())
            if x[0].strip() != ''
        }

    if vocab is not None:

        extras = {x.norm_: substitutions[x.lower_] for x in vocab if x.lower_ in substitutions}
        substitutions.update(extras)

        extras = {x.lower_: substitutions[x.norm_] for x in vocab if x.norm_ in substitutions}
        substitutions.update(extras)

    substitutions = {k: v for k, v in substitutions.items() if k != v}

    return substitutions


def read_yaml(file: Any) -> dict:
    """Read yaml file. Return dict."""
    if isinstance(file, str) and any(file.endswith(x) for x in ('.yml', '.yaml')):
        with open(file, "r", encoding='utf-8') as fp:
            return yaml.load(fp, Loader=yaml.FullLoader)
    data: List[dict] = yaml.load(file, Loader=yaml.FullLoader)
    return {} if len(data) == 0 else data[0]


def write_yaml(data: dict, file: str) -> None:
    """Write yaml to file.."""
    with open(file, "w", encoding='utf-8') as fp:
        return yaml.dump(data=data, stream=fp)


def update_dict_from_yaml(yaml_file: str, data: dict) -> dict:
    """Update dict `data` with values found in `yaml_file`."""
    if yaml_file is None:
        return data
    options: dict = read_yaml(yaml_file)
    data.update(options)
    return data
