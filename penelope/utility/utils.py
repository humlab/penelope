# -*- coding: utf-8 -*-
import datetime
import functools
import glob
import inspect
import itertools
import json
import logging
import os
import platform
import re
import string
import time
import zipfile
from typing import List, Tuple

import gensim.utils
import numpy as np
import pandas as pd


def setup_logger(
    logger=None, to_file=False, filename=None, level=logging.DEBUG
):  # pylint: disable=redefined-outer-name
    """
    Setup logging of import messages to both file and console
    """
    if logger is None:
        logger = logging.getLogger("")

    logger.handlers = []

    logger.setLevel(level)
    formatter = logging.Formatter('%(message)s')

    if to_file is True or filename is not None:
        if filename is None:
            filename = '_{}.log'.format(time.strftime("%Y%m%d"))
        fh = logging.FileHandler(filename)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def getLogger(name='', level=logging.INFO):
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=level)
    _logger = logging.getLogger(name)
    _logger.setLevel(level)
    return _logger


logger = getLogger(__name__)

lazy_flatten = gensim.utils.lazy_flatten
iter_windows = gensim.utils.iter_windows
deprecated = gensim.utils.deprecated


def remove_snake_case(snake_str):
    return ' '.join(x.title() for x in snake_str.split('_'))


def noop(*args):  # pylint: disable=unused-argument
    pass


def isint(s):
    try:
        int(s)
        return True
    except:  # pylint: disable=bare-except
        return False


def filter_dict(d, keys=None, filter_out=False):
    keys = set(d.keys()) - set(keys or []) if filter_out else (keys or [])
    return {k: v for k, v in d.items() if k in keys}


def timecall(f):
    @functools.wraps(f)
    def f_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = f(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        logger.info("Call time [{}]: {:.4f} secs".format(f.__name__, elapsed))
        return value

    return f_wrapper


def extend(target, *args, **kwargs):
    """Returns dictionary 'target' extended by supplied dictionaries (args) or named keywords

    Parameters
    ----------
    target : dict
        Default dictionary (to be extended)

    args: [dict]
        Optional. List of dicts to use when updating target

    args: [key=value]
        Optional. List of key-value pairs to use when updating target

    Returns
    -------
    [dict]
        Target dict updated with supplied dicts/key-values.
        Multiple keys are overwritten inorder of occrence i.e. keys to right have higher precedence

    """

    target = dict(target)
    for source in args:
        target.update(source)
    target.update(kwargs)
    return target


def ifextend(target, source, p):
    return extend(target, source) if p else target


def extend_single(target, source, name):
    if name in source:
        target[name] = source[name]
    return target


def flatten(lofl):
    """Returns a flat single list out of supplied list of lists."""

    return [item for sublist in lofl for item in sublist]


def project_series_to_range(series, low, high):
    """Project a sequence of elements to a range defined by (low, high)"""
    norm_series = series / series.max()
    return norm_series.apply(lambda x: low + (high - low) * x)


def project_to_range(value, low, high):
    """Project a singlevalue to a range (low, high)"""
    return low + (high - low) * value


def clamp_values(values, low_high):
    """Clamps value to supplied interval."""
    mw = max(values)
    return [project_to_range(w / mw, low_high[0], low_high[1]) for w in values]


@functools.lru_cache(maxsize=512)
def _get_signature(func):
    return inspect.signature(func)


def get_func_args(func):
    sig = _get_signature(func)
    return [
        arg_name for arg_name, param in sig.parameters.items() if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]


def filter_kwargs(f, args):
    """Removes keys in dict arg that are invalid arguments to function f

    Parameters
    ----------
    f : [fn]
        Function to introspect
    args : dict
        List of parameter names to test validity of.

    Returns
    -------
    dict
        Dict with invalid args filtered out.
    """

    try:
        return {k: args[k] for k in args.keys() if k in get_func_args(f)}

    except:  # pylint: disable=bare-except
        return args


def inspect_filter_args(f, args):
    return {k: args[k] for k in args.keys() if k in inspect.getfullargspec(f).args}


def inspect_default_opts(f):
    sig = inspect.signature(f)
    return {name: param.default for name, param in sig.parameters.items() if param.name != 'self'}


VALID_CHARS = "-_.() " + string.ascii_letters + string.digits


def filename_whitelist(filename):
    filename = ''.join(x for x in filename if x in VALID_CHARS)
    return filename


def dict_subset(d, keys):
    if keys is None:
        return d
    return {k: v for (k, v) in d.items() if k in keys}


def dict_split(d, fn):
    """Splits a dictionary into two parts based on predicate """
    true_keys = {k for k in d.keys() if fn(d, k)}
    return {k: d[k] for k in true_keys}, {k: d[k] for k in set(d.keys()) - true_keys}


def list_of_dicts_to_dict_of_lists(dl):
    dict_of_lists = dict(zip(dl[0], zip(*[d.values() for d in dl])))
    return dict_of_lists


def tuple_of_lists_to_list_of_tuples(tl):
    return zip(*tl)


def dict_of_lists_to_list_of_dicts(dl):
    return [dict(zip(dl, t)) for t in zip(*dl.values())]


def uniquify(sequence):
    """ Removes duplicates from a list whilst still preserving order """
    seen = set()
    seen_add = seen.add
    return [x for x in sequence if not (x in seen or seen_add(x))]


sort_chained = lambda x, f: list(x).sort(key=f) or x


def ls_sorted(path):
    return sort_chained(list(filter(os.path.isfile, glob.glob(path))), os.path.getmtime)


def split(delimiters, text, maxsplit=0):
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, text, maxsplit)


def path_add_suffix(path, suffix, new_extension=None):
    basename, extension = os.path.splitext(path)
    suffixed_path = basename + suffix + (extension if new_extension is None else new_extension)
    return suffixed_path


def path_add_timestamp(path, fmt="%Y%m%d%H%M"):
    suffix = '_{}'.format(time.strftime(fmt))
    return path_add_suffix(path, suffix)


def path_add_date(path, fmt="%Y%m%d"):
    suffix = '_{}'.format(time.strftime(fmt))
    return path_add_suffix(path, suffix)


def path_add_sequence(path, i, j=0):
    suffix = str(i).zfill(j)
    return path_add_suffix(path, suffix)


def zip_get_filenames(zip_filename, extension='.txt'):
    with zipfile.ZipFile(zip_filename, mode='r') as zf:
        return [x for x in zf.namelist() if x.endswith(extension)]


def zip_get_text(zip_filename, filename):
    with zipfile.ZipFile(zip_filename, mode='r') as zf:
        return zf.read(filename).decode(encoding='utf-8')


def slim_title(x):
    try:
        m = re.match(r'.*\((.*)\)$', x).groups()
        if m is not None and len(m) > 0:
            return m[0]
        return ' '.join(x.split(' ')[:3]) + '...'
    except:  # pylint: disable=bare-except
        return x


def complete_value_range(values, typef=str):
    """Create a complete range from min/max range in case values are missing

    Parameters
    ----------
    str_values : list
        List of values to fill

    Returns
    -------
    """

    if len(values) == 0:
        return []

    values = list(map(int, values))
    values = range(min(values), max(values) + 1)

    return list(map(typef, values))


def is_platform_architecture(xxbit):
    assert xxbit in ['32bit', '64bit']
    logger.info(platform.architecture()[0])
    return platform.architecture()[0] == xxbit
    # return xxbit == ('64bit' if sys.maxsize > 2**32 else '32bit')


def trunc_year_by(series, divisor):
    return (series - series.mod(divisor)).astype(int)


# FIXA! Use numpy instead
def normalize_values(values):
    if len(values or []) == 0:
        return []
    max_value = max(values)
    if max_value == 0:
        return values
    values = [x / max_value for x in values]
    return values


def normalize_array(x: np.ndarray, ord: int = 1):  # pylint: disable=redefined-builtin
    """
    function that normalizes an ndarray of dim 1d

    Args:
     ``x``: A numpy array

    Returns:
     ``x``: The normalize darray.
    """
    norm = np.linalg.norm(x, ord=ord)
    return x / (norm if norm != 0 else 1.0)


def extract_counter_items_within_threshold(counter, low, high):
    item_values = set([])
    for x, wl in counter.items():
        if low <= x <= high:
            item_values.update(wl)
    return item_values


def chunks(lst, n):

    if (n or 0) == 0:
        yield lst

    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# def get_document_id_by_field_filters(documents, filters):
#     df = documents
#     for k, v in filters:
#         if len(v or []) > 0:
#             df = df[df[k].isin(v)]
#     return list(df.index)

# def get_documents_by_field_filters(corpus, documents, filters):
#     ids = get_document_id_by_field_filters(documents, filters)
#     docs = ( x for x in corpus if x._.meta['document_id'] in ids)
#     return docs

# def get_tagset(data_folder, filename='tagset.csv'):
#     filepath = os.path.join(data_folder, filename)
#     if os.path.isfile(filepath):
#         return pd.read_csv(filepath, sep='\t').fillna('')
#     return None

# def pos_tags(data_folder, filename='tagset.csv'):
#     df_tagset = pd.read_csv(os.path.join(data_folder, filename), sep='\t').fillna('')
#     return df_tagset.groupby(['POS'])['DESCRIPTION'].apply(list).apply(lambda x: ', '.join(x[:1])).to_dict()


def dataframe_to_tuples(df: pd.DataFrame, columns: List[str] = None) -> List[Tuple]:
    """Returns rows in dataframe as tuples"""
    if columns is not None:
        df = df[columns]
    tuples = [tuple(x.values()) for x in df.to_dict(orient='index').values()]
    return tuples


def nth(iterable, n: int, default=None):
    "Returns the nth item or a default value"
    return next(itertools.islice(iterable, n, None), default)


def read_json(path):
    with open(path) as fp:
        return json.load(fp)


def now_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S')


def timestamp(format_string=None):
    """ Add timestamp to string that must contain exacly one placeholder """
    tz = now_timestamp()
    return tz if format_string is None else format_string.format(tz)


def suffix_filename(filename, suffix):
    output_path, output_file = os.path.split(filename)
    output_base, output_ext = os.path.splitext(output_file)
    suffixed_filename = os.path.join(output_path, f"{output_base}_{suffix}{output_ext}")
    return suffixed_filename


def replace_extension(filename, extension):
    if filename.endswith(extension):
        return filename
    base, _ = os.path.splitext(filename)
    return f"{base}{'' if extension.startswith('.') else '.'}{extension}"


def timestamp_filename(filename):
    return suffix_filename(filename, now_timestamp())


def project_values_to_range(values, low, high):
    w_max = max(values)
    return [low + (high - low) * (x / w_max) for x in values]


# HYPHEN_REGEXP = re.compile(r'\b(\w+)-\s*\r?\n\s*(\w+)\b', re.UNICODE)

# def fix_hyphenation(text: str) -> str:
#     result = re.sub(HYPHEN_REGEXP, r"\1\2\n", text)
#     return result

# def fix_whitespaces(text: str) -> str:
#     result = re.sub(r'\s+', ' ', text).strip()
#     return result
