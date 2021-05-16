import zipfile
from os.path import isfile
from typing import AnyStr, Callable, Iterable, List, Tuple, Union

from penelope.utility import streamify_any_source

from .interfaces import TextSource
from .text_reader import TextReaderOpts
from .zip_iterator import ZipTextIterator


# pylint: disable=too-many-return-statements
def streamify_text_source(
    text_source: TextSource,
    *,
    filename_pattern: str = '*.txt',
    filename_filter: Union[List[str], Callable] = None,
    as_binary: bool = False,
    n_processes: int = 1,
    n_chunksize: int = 1,
) -> Iterable[Tuple[str, AnyStr]]:
    """Returns an (filename, text) iterator for `text_source`

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

    if isfile(text_source) and zipfile.is_zipfile(text_source):
        return ZipTextIterator(
            text_source,
            reader_opts=TextReaderOpts(
                filename_pattern=filename_pattern,
                filename_filter=filename_filter,
                as_binary=as_binary,
                n_processes=n_processes,
                n_chunksize=n_chunksize,
            ),
        )
    return streamify_any_source(
        source=text_source,
        filename_pattern=filename_pattern,
        filename_filter=filename_filter,
        as_binary=as_binary,
        n_processes=n_processes,
        n_chunksize=n_chunksize,
    )
