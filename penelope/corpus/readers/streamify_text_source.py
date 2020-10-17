import glob
import os
import zipfile
from typing import Any, Callable, List, Union

import penelope.utility.file_utility as file_utility

from .zip_iterator import ZipTextIterator


def streamify_text_source(
    text_source: Any,
    filename_pattern: str = '*.txt',
    filename_filter: Union[List[str], Callable] = None,
    as_binary: bool = False,
):
    """Returns an (file_pattern, text) iterator for `text_source`

    Parameters
    ----------
    text_source : Union[str,List[(str,str)]]
        Filename, folder name or an iterator that returns a (filename, text) stream
    file_pattern : str, optional
        Filter for file exclusion, a patter or a predicate, by default '*.txt'
    as_binary : bool, optional
        Read tex as binary (unicode) data, by default False

    Returns
    -------
    Iterable[Tuple[str,str]]
        A stream of filename, text tuples
    """

    if not isinstance(text_source, str):
        if hasattr(text_source, '__iter__') and hasattr(text_source, '__next__'):
            return text_source

    if isinstance(text_source, list):

        if len(text_source) == 0:
            return []

        if isinstance(text_source[0], tuple):
            return text_source

        return ((f'document_{i+1}.txt', d) for i, d in enumerate(text_source))

    if os.path.isfile(text_source):

        if zipfile.is_zipfile(text_source):
            return ZipTextIterator(
                text_source, filename_pattern=filename_pattern, filename_filter=filename_filter, as_binary=as_binary
            )

        text = file_utility.read_textfile(text_source, as_binary=as_binary)
        return ((text_source, text),)

    if os.path.isdir(text_source):

        return (
            (os.path.basename(filename), file_utility.read_textfile(filename, as_binary=as_binary))
            for filename in glob.glob(os.path.join(text_source, filename_pattern))
            if file_utility.filename_satisfied_by(os.path.basename(filename), filename_filter)
        )

    return (('document', x) for x in [text_source])
