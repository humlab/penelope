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
from numbers import Number
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Sequence, Set, Tuple, TypeVar

import gensim.utils
import numpy as np
import pandas as pd

T = TypeVar('T')


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


def getLogger(name: str = '', level=logging.INFO):
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=level)
    _logger = logging.getLogger(name)
    _logger.setLevel(level)
    return _logger


logger = getLogger(__name__)

lazy_flatten = gensim.utils.lazy_flatten
iter_windows = gensim.utils.iter_windows
deprecated = gensim.utils.deprecated


def remove_snake_case(snake_str: str) -> str:
    return ' '.join(x.title() for x in snake_str.split('_'))


def noop(*_):
    pass


def isint(s: Any) -> bool:
    try:
        int(s)
        return True
    except:  # pylint: disable=bare-except
        return False


def filter_dict(d: Dict[str, Any], keys: Sequence[str] = None, filter_out: bool = False) -> Dict[str, Any]:
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


def extend(target: Mapping, *args, **kwargs) -> Mapping:
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


def ifextend(target: Mapping, source: Mapping, p: bool) -> Mapping:
    return extend(target, source) if p else target


def extend_single(target: Mapping, source: Mapping, name: str) -> Mapping:
    if name in source:
        target[name] = source[name]
    return target


def flatten(lofl: List[List[T]]) -> List[T]:
    """Returns a flat single list out of supplied list of lists."""

    return [item for sublist in lofl for item in sublist]


def project_series_to_range(series: Sequence[Number], low: Number, high: Number) -> Sequence[Number]:
    """Project a sequence of elements to a range defined by (low, high)"""
    norm_series = series / series.max()
    return norm_series.apply(lambda x: low + (high - low) * x)


def project_to_range(value: Sequence[Number], low: Number, high: Number) -> Sequence[Number]:
    """Project a singlevalue to a range (low, high)"""
    return low + (high - low) * value


def clamp_values(values: Sequence[Number], low_high: Tuple[Number, Number]) -> Sequence[Number]:
    """Clamps value to supplied interval."""
    mw = max(values)
    return [project_to_range(w / mw, low_high[0], low_high[1]) for w in values]


@functools.lru_cache(maxsize=512)
def _get_signature(func: Callable) -> inspect.Signature:
    return inspect.signature(func)


def get_func_args(func: Callable) -> List[str]:
    sig = _get_signature(func)
    return [
        arg_name for arg_name, param in sig.parameters.items() if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]


def filter_kwargs(f: Callable, args: Mapping[str, Any]) -> Mapping[str, Any]:
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


def inspect_filter_args(f: Callable, args: Mapping) -> Mapping:
    return {k: args[k] for k in args.keys() if k in inspect.getfullargspec(f).args}


def inspect_default_opts(f: Callable) -> Mapping:
    sig = inspect.signature(f)
    return {name: param.default for name, param in sig.parameters.items() if param.name != 'self'}


VALID_CHARS = "-_.() " + string.ascii_letters + string.digits


def filename_whitelist(filename: str) -> str:
    filename = ''.join(x for x in filename if x in VALID_CHARS)
    return filename


def dict_subset(d: Mapping, keys: Sequence[str]) -> Mapping:
    if keys is None:
        return d
    return {k: v for (k, v) in d.items() if k in keys}


def dict_split(d: Mapping, fn: Callable[[Mapping, str], bool]) -> Mapping:
    """Splits a dictionary into two parts based on predicate """
    true_keys = {k for k in d.keys() if fn(d, k)}
    return {k: d[k] for k in true_keys}, {k: d[k] for k in set(d.keys()) - true_keys}


def list_of_dicts_to_dict_of_lists(dl: List[Mapping[str, Any]]) -> Mapping[str, List[Any]]:
    dict_of_lists = dict(zip(dl[0], zip(*[d.values() for d in dl])))
    return dict_of_lists


def tuple_of_lists_to_list_of_tuples(tl: Tuple[List[Any], ...]) -> List[Tuple[Any, ...]]:
    return zip(*tl)


def dict_of_lists_to_list_of_dicts(dl: Mapping[str, List[Any]]) -> List[Mapping[str, Any]]:
    return [dict(zip(dl, t)) for t in zip(*dl.values())]


ListOfDicts = List[Mapping[str, Any]]


def lists_of_dicts_merged_by_key(lst1: ListOfDicts, lst2: ListOfDicts, key: str) -> ListOfDicts:
    """Returns `lst1` where each items has been merged with corresponding item in `lst2` using common field `key`"""
    if lst2 is None or len(lst2) == 0 or key not in lst2[0]:
        return lst1 or []

    if lst1 is None:
        return None

    if len(lst1) > 0 and key not in lst1[0]:
        raise ValueError(f"Key `{key}` not in target list")

    lookup = {item[key]: item for item in lst2}
    merged_list = map(lambda x: {**x, **lookup.get(x[key], {})}, lst1)

    return list(merged_list)


def uniquify(sequence: Iterable[T]) -> List[T]:
    """ Removes duplicates from a list whilst still preserving order """
    seen = set()
    seen_add = seen.add
    return [x for x in sequence if not (x in seen or seen_add(x))]


def sort_chained(x, f):
    return list(x).sort(key=f) or x


def ls_sorted(path: str) -> List[str]:
    return sort_chained(list(filter(os.path.isfile, glob.glob(path))), os.path.getmtime)


def split(delimiters: Sequence[str], text: str, maxsplit: int = 0) -> List[str]:
    reg_ex = '|'.join(map(re.escape, delimiters))
    return re.split(reg_ex, text, maxsplit)


def path_add_suffix(path: str, suffix: str, new_extension: str = None) -> str:
    basename, extension = os.path.splitext(path)
    suffixed_path = basename + suffix + (extension if new_extension is None else new_extension)
    return suffixed_path


def path_add_timestamp(path: str, fmt: str = "%Y%m%d%H%M") -> str:
    suffix = '_{}'.format(time.strftime(fmt))
    return path_add_suffix(path, suffix)


def path_add_date(path: str, fmt: str = "%Y%m%d") -> str:
    suffix = '_{}'.format(time.strftime(fmt))
    return path_add_suffix(path, suffix)


def path_add_sequence(path: str, i: int, j: int = 0) -> str:
    suffix = str(i).zfill(j)
    return path_add_suffix(path, suffix)


def zip_get_filenames(zip_filename: str, extension: str = '.txt') -> List[str]:
    with zipfile.ZipFile(zip_filename, mode='r') as zf:
        return [x for x in zf.namelist() if x.endswith(extension)]


def zip_get_text(zip_filename: str, filename: str) -> str:
    with zipfile.ZipFile(zip_filename, mode='r') as zf:
        return zf.read(filename).decode(encoding='utf-8')


def right_chop(s: str, suffix: str) -> str:
    """Returns `s` with `suffix` removed"""
    return s[: -len(suffix)] if suffix != "" and s.endswith(suffix) else s


def left_chop(s: str, suffix: str) -> str:
    """Returns `s` with `suffix` removed"""
    return s[len(suffix) :] if suffix != "" and s.startswith(suffix) else s


def slim_title(x: str) -> str:
    try:
        m = re.match(r'.*\((.*)\)$', x).groups()
        if m is not None and len(m) > 0:
            return m[0]
        return ' '.join(x.split(' ')[:3]) + '...'
    except:  # pylint: disable=bare-except
        return x


def complete_value_range(values: Sequence[Number], typef=str) -> List[Number]:
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


def is_platform_architecture(xxbit: str) -> bool:
    assert xxbit in ['32bit', '64bit']
    logger.info(platform.architecture()[0])
    return platform.architecture()[0] == xxbit
    # return xxbit == ('64bit' if sys.maxsize > 2**32 else '32bit')


def trunc_year_by(series, divisor):
    return (series - series.mod(divisor)).astype(int)


# FIXA! Use numpy instead
def normalize_values(values: Sequence[Number]) -> Sequence[Number]:
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


def extract_counter_items_within_threshold(counter: Mapping, low: Number, high: Number) -> Set:
    item_values = set([])
    for x, wl in counter.items():
        if low <= x <= high:
            item_values.update(wl)
    return item_values


def chunks(lst: List[T], n: int) -> List[T]:

    if (n or 0) == 0:
        yield lst

    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def dataframe_to_tuples(df: pd.DataFrame, columns: List[str] = None) -> List[Tuple]:
    """Returns rows in dataframe as tuples"""
    if columns is not None:
        df = df[columns]
    tuples = [tuple(x.values()) for x in df.to_dict(orient='index').values()]
    return tuples


def nth(iterable: Iterable[T], n: int, default: T = None) -> T:
    "Returns the nth item or a default value"
    return next(itertools.islice(iterable, n, None), default)


def take(n: int, iterable: Iterator):
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, n))


def read_json(path: str) -> Any:
    with open(path) as fp:
        return json.load(fp)


def now_timestamp() -> str:
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S')


def timestamp(format_string: str = None) -> str:
    """ Add timestamp to string that must contain exacly one placeholder """
    tz = now_timestamp()
    return tz if format_string is None else format_string.format(tz)


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


def project_values_to_range(values: List[Number], low: Number, high: Number) -> List[Number]:
    w_max = max(values)
    return [low + (high - low) * (x / w_max) for x in values]


def pretty_print_matrix(
    M, row_labels: List[str], column_labels: List[str], dtype: type = np.float64, float_fmt: str = "{0:.04f}"
):
    """Pretty-print a matrix using Pandas."""
    df = pd.DataFrame(M, index=row_labels, columns=column_labels, dtype=dtype)
    if issubclass(np.float64, np.floating):
        with pd.option_context('float_format', float_fmt.format):
            print(df)
    else:
        print(df)
