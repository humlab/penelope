import datetime
import functools
import glob
import inspect
import itertools
import logging
import os
import platform
import re
import time
import uuid
from importlib import import_module
from numbers import Number
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Sequence, Set, Tuple, Type, TypeVar, Union

import gensim.utils
import numpy as np
import pandas as pd
import scipy

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


LOG_FORMAT = "%(asctime)s : %(levelname)s : %(message)s"


def fn_name(default=None):
    try:
        return inspect.stack()[1][3]
    except Exception:
        return default or str(uuid.uuid1())


def get_logger(
    name: str = "penelope",
    *,
    to_file: Union[bool, str] = False,
    level: int = logging.DEBUG,
):  # pylint: disable=redefined-outer-name
    """
    Setup logging of messages to both file and console
    """

    logger = getLogger(name, level=level)

    if to_file and isinstance(to_file, (bool, str)):
        fh = logging.FileHandler(f'{name}_{time.strftime("%Y%m%d")}.log' if isinstance(to_file, bool) else to_file)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt=LOG_FORMAT))
        logger.addHandler(fh)

    return logger


def getLogger(name: str = '', level=logging.INFO):
    logging.basicConfig(format=LOG_FORMAT, level=level)
    _logger = logging.getLogger(name)
    _logger.setLevel(level)
    return _logger


logger = getLogger(__name__)

lazy_flatten = gensim.utils.lazy_flatten
iter_windows = gensim.utils.iter_windows
deprecated = gensim.utils.deprecated


def to_text(data: Union[str, Iterable[str]]):
    return data if isinstance(data, str) else ' '.join(data)


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


def better_flatten2(lst) -> Iterable[Any]:
    for el in lst:
        if isinstance(el, (Iterable,)) and not isinstance(  # pylint: disable=isinstance-second-argument-not-valid-type
            el, (str, bytes)
        ):
            yield from better_flatten2(el)
        else:
            yield el


def better_flatten(lst: Iterable[Any]) -> List[Any]:
    if isinstance(lst, (str, bytes)):
        return lst
    return [x for x in better_flatten2(lst)]


def project_series_to_range(series: Sequence[Number], low: Number, high: Number) -> Sequence[Number]:
    """Project a sequence of elements to a range defined by (low, high)"""
    norm_series = series / series.max()
    return norm_series.apply(lambda x: low + (high - low) * x)


def project_values_to_range(values: List[Number], low: Number, high: Number) -> List[Number]:
    w_max = max(values)
    return [low + (high - low) * (x / w_max) for x in values]


def project_to_range(value: Sequence[Number], low: Number, high: Number) -> Sequence[Number]:
    """Project a singlevalue to a range (low, high)"""
    return low + (high - low) * value


def clamp_values(values: Sequence[Number], low_high: Tuple[Number, Number]) -> Sequence[Number]:
    """Clamps value to supplied interval."""
    if not values:
        return values
    mw = max(values)
    return [project_to_range(w / mw, low_high[0], low_high[1]) for w in values]


def clamp(n: int, smallest: int, largest: int) -> int:
    """Clamps integers to a range"""
    return max(smallest, min(n, largest))


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


def dict_subset(d: Mapping, keys: Sequence[str]) -> Mapping:
    if keys is None:
        return d
    return {k: v for (k, v) in d.items() if k in keys}


def dict_split(d: Mapping, fn: Callable[[Mapping, str], bool]) -> Mapping:
    """Splits a dictionary into two parts based on predicate """
    true_keys = {k for k in d.keys() if fn(d, k)}
    return {k: d[k] for k in true_keys}, {k: d[k] for k in set(d.keys()) - true_keys}


def dict_to_list_of_tuples(d: Mapping) -> List[Tuple[Any, Any]]:
    if d is None:
        return []
    return [(k, v) for (k, v) in d.items()]


def list_of_dicts_to_dict_of_lists(dl: List[Mapping[str, Any]]) -> Mapping[str, List[Any]]:
    dict_of_lists = dict(zip(dl[0], zip(*[d.values() for d in dl])))
    return dict_of_lists


def tuple_of_lists_to_list_of_tuples(tl: Tuple[List[Any], ...]) -> List[Tuple[Any, ...]]:
    return zip(*tl)


def dict_of_lists_to_list_of_dicts(dl: Mapping[str, List[Any]]) -> List[Mapping[str, Any]]:
    return [dict(zip(dl, t)) for t in zip(*dl.values())]


def dict_of_key_values_inverted_to_dict_of_value_key(d: Dict[K, List[V]]) -> Dict[V, K]:
    return {value: key for key in d for value in d[key]}


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


def list_to_unique_list_with_preserved_order(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def uniquify(sequence: Iterable[T]) -> List[T]:
    """ Removes duplicates from a list whilst still preserving order """
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def sort_chained(x, f):
    return list(x).sort(key=f) or x


def ls_sorted(path: str) -> List[str]:
    return sort_chained(list(filter(os.path.isfile, glob.glob(path))), os.path.getmtime)


def split(delimiters: Sequence[str], text: str, maxsplit: int = 0) -> List[str]:
    reg_ex = '|'.join(map(re.escape, delimiters))
    return re.split(reg_ex, text, maxsplit)


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


def now_timestamp() -> str:
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S')


def timestamp(format_string: str = None) -> str:
    """ Add timestamp to string that must contain exacly one placeholder """
    tz = now_timestamp()
    return tz if format_string is None else format_string.format(tz)


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


def assert_is_strictly_increasing(series: pd.Series):
    """[summary]

    Args:
        series (pd.Series): [description]

    Raises:
        ValueError: [description]
    """
    if not is_strictly_increasing(series):
        raise ValueError(f"series: {series.name} must be an integer typed, strictly increasing series starting from 0")


def is_strictly_increasing(series: pd.Series, by_value=1, start_value: int = 0, sort_values: bool = True):

    if len(series) == 0:
        return True

    if not np.issubdtype(series.dtype, np.integer):
        return False

    if sort_values:
        series = series.sort_values()

    if start_value is not None:
        if series[0] != start_value:
            return False

    if not series.is_monotonic_increasing:
        return False

    if by_value is not None:
        if not np.all((series[1:].values - series[:-1].values) == by_value):
            return False

    return True


def normalize_sparse_matrix_by_vector(spm: scipy.sparse.spmatrix, vector: np.ndarray = None) -> scipy.sparse.spmatrix:
    # https://stackoverflow.com/questions/42225269/scipy-sparse-matrix-division
    # diagonal matrix from the reciprocals of vector x sparse matrix
    vector = vector if vector is not None else spm.sum(axis=1).A1
    nspm = scipy.sparse.diags(1.0 / vector) @ spm
    nspm.data[(np.isnan(nspm.data) | np.isposinf(nspm.data))] = 0.0
    return nspm


# def sparse_normalize(spm: scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
#     # https://stackoverflow.com/questions/42225269/scipy-sparse-matrix-division
#     row_sums = spm.sum(axis=1).A1
#     # diagonal matrix from the reciprocals of row sums:
#     row_sum_reciprocals_diagonal = scipy.sparse.diags(1. / row_sums)
#     nspm = row_sum_reciprocals_diagonal @ spm
#     nspm.data[(np.isnan(nspm.data)|np.isposinf(nspm.data))] = 0.0
#     return nspm


class DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def create_instance(class_or_function_path: str) -> Union[Callable, Type]:
    try:
        module_path, cls_or_function_name = class_or_function_path.rsplit('.', 1)
        module = import_module(module_path)
        return getattr(module, cls_or_function_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(f"fatal: config error: unable to load {class_or_function_path}") from e


def multiple_replace(text: str, replace_map: dict, ignore_case: bool = False) -> str:
    # Create a regular expression  from the dictionary keys
    opts = dict(flags=re.IGNORECASE) if ignore_case else {}
    sorted_keys = sorted(replace_map.keys(), key=lambda k: len(replace_map[k]), reverse=True)
    regex = re.compile(f"({'|'.join(map(re.escape, sorted_keys))})", **opts)
    if ignore_case:
        fx = lambda mo: replace_map[(mo.string[mo.start() : mo.end()]).lower()]
    else:
        fx = lambda mo: replace_map[mo.string[mo.start() : mo.end()]]
    return regex.sub(fx, text)
