import os
from fnmatch import fnmatch
from glob import glob
from typing import Any, List, Optional
from zipfile import ZipFile

import pandas as pd
from penelope.utility import extract_filenames_metadata, filter_names_by_pattern, getLogger, read_textfile, strip_paths

from ..interfaces import TextReaderOpts
from .interfaces import ISource, SourceInfo, StoreItemPair

logger = getLogger("penelope")


class ZipSource(ISource):
    def __init__(self, source_path: str):

        self.source_path = source_path

        self.zipfile = None
        self.enter_depth = 0

        with self:
            self.filenames = list(sorted(self.zipfile.namelist()))

    def namelist(self, *, pattern: str = '*.*') -> List[str]:
        return filter_names_by_pattern(self.filenames, pattern)

    def read(self, filename: str, as_binary: bool = False) -> Optional[Any]:
        with self:
            with self.zipfile.open(filename, 'r') as fp:
                content = fp.read() if as_binary else fp.read().decode('utf-8')
        return content

    def exists(self, filename: str) -> bool:
        return filename in self.filenames

    def __enter__(self):  # pylint: disable=unused-argument
        if self.zipfile is None:
            self.zipfile = ZipFile(self.source_path, "r")  # pylint: disable=consider-using-with
            self.enter_depth = 1
        else:
            self.enter_depth += 1
        return self

    def __exit__(self, _, value, traceback):  # pylint: disable=unused-argument
        self.enter_depth -= 1
        if self.zipfile and self.enter_depth == 0:
            self.zipfile.close()
            self.zipfile = None


class FolderSource(ISource):
    def __init__(self, source_path: str):
        self.source_path = source_path

    def to_path(self, filename: str) -> str:
        return os.path.join(self.source_path, filename)

    def namelist(self, *, pattern: str = '*.*') -> List[str]:
        return strip_paths(glob(self.to_path(pattern or '*.*')))

    def read(self, filename: str, as_binary: bool = False) -> Optional[Any]:
        return read_textfile(self.to_path(filename), as_binary=as_binary)

    def exists(self, filename: str) -> bool:
        return os.path.isfile(self.to_path(filename))

    def __exit__(self, _, value, traceback):  # pylint: disable=unused-argument
        ...


class InMemorySource(ISource):
    def __init__(self, items: List[StoreItemPair]):
        self.items: List[StoreItemPair] = items
        self.map = {strip_paths(item[0]): item for item in self.items}
        self.filenames = strip_paths([name for name, _ in self.items])

    def namelist(self, *, pattern: str = '*.*') -> List[str]:  # pylint: disable=unused-argument
        return strip_paths([name for name, _ in self.items if fnmatch(name, pattern)])

    def read(self, filename: str, as_binary: bool = False) -> Optional[Any]:  # pylint: disable=unused-argument
        return self.map[filename][1]

    def exists(self, filename: str) -> bool:
        return filename in self.filenames

    def __exit__(self, _, value, traceback):  # pylint: disable=unused-argument
        ...


class PandasSource(ISource):  # pylint: disable=too-many-instance-attributes, disable=too-many-return-statements
    """Text iterator that returns row-wise text documents from a Pandas DataFrame"""

    def __init__(self, data: pd.DataFrame, text_column='txt', filename_column='filename', **column_filters):

        self.data: pd.DataFrame = data
        self.text_column: str = text_column
        self.column_filters = None

        if 'filename' not in self.data.columns:
            self.data['filename'] = (
                self.data[filename_column].astype(str)
                if filename_column in self.data.columns
                else self.data.index.to_series().apply(lambda z: f"document_{z}.txt")
            )

        if len(self.data[self.data[text_column].isna()]) > 0:
            logger('Warn: {} n/a rows encountered'.format(len(self.data[self.data[text_column].isna()])))
            self.data = self.data.dropna()

        self.data: pd.DataFrame = self.data.set_index('filename', drop=False).sort_index().rename_axis('')
        self.filtered_data = data
        self.apply_filters(**column_filters)

    def namelist(self, *, pattern: str = '*') -> List[str]:  # pylint: disable=unused-argument
        return [x for x in self.filtered_data.filename.tolist() if fnmatch(x, pattern)]

    def read(self, filename: str, as_binary: bool = False) -> Optional[Any]:  # pylint: disable=unused-argument
        return str(self.filtered_data.loc[filename][self.text_column])

    def exists(self, filename: str) -> bool:
        return filename in self.filtered_data.index

    def get_info(self, opts: TextReaderOpts) -> SourceInfo:

        filenames = self.namelist(pattern=opts.filename_pattern)
        basenames = strip_paths(filenames)
        filename_metadata = extract_filenames_metadata(filenames=basenames, filename_fields=opts.filename_fields)
        columns = [x for x in self.filtered_data.columns.tolist() if x != self.text_column]
        dataframe_metadata = self.filtered_data[columns].to_dict('records')
        metadata = [{**x, **y} for x, y in zip(filename_metadata, dataframe_metadata)]
        name_to_filename = {strip_paths(name): filename for name, filename in zip(basenames, filenames)}

        return SourceInfo(name_to_filename=name_to_filename, names=basenames, metadata=metadata)

    def apply_filters(self, **column_filters) -> "PandasSource":
        self.column_filters = column_filters
        df = self.data
        for column, value in (column_filters or dict()).items():
            df = (
                df[df[column].between(*value)]
                if isinstance(value, tuple)
                else df[df[column].isin(value)]
                if isinstance(value, list)
                else df[df[column] == value]
            )
        self.filtered_data = df
        return self

    def __exit__(self, _, value, traceback):  # pylint: disable=unused-argument
        ...
