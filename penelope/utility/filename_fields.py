import logging
import os
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from .filename_utils import strip_paths

FilenameFieldSpec = Union[List[str], Dict[str, Union[Callable, str]]]
FilenameFieldSpecs = Optional[Sequence[FilenameFieldSpec]]
NameFieldSpecs = Optional[FilenameFieldSpecs]


def _parse_indexed_fields(filename_fields: List[str]):
    """Parses a list of meta-field expressions into a format suitable for `extract_filename_fields`
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

    except Exception as ex:  # pylint: disable=bare-except
        logging.exception(ex)
        print("parse error: meta-fields, must be in format 'name:regexp'")
        raise


def extract_filename_metadata(filename: str, filename_fields: FilenameFieldSpec) -> Mapping[str, Any]:
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
        filename_fields = _parse_indexed_fields(filename_fields)

    if isinstance(filename_fields, str):
        # List of `key:sep:index`
        filename_fields = _parse_indexed_fields(filename_fields.split('#'))

    key_fx = {key: fxify(fx_or_re) for key, fx_or_re in filename_fields.items()}

    data = {'filename': basename}
    for key, fx in key_fx.items():
        data[key] = astype_int_or_str(fx(basename))

    return data


def extract_filenames_metadata(*, filenames: List[str], filename_fields: FilenameFieldSpecs) -> List[Mapping[str, Any]]:
    return [
        {'filename': filename, **extract_filename_metadata(filename, filename_fields)}
        for filename in strip_paths(filenames)
    ]
