import abc
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from penelope.utility import extract_filenames_metadata, filename_satisfied_by, strip_paths

from ..interfaces import TextReaderOpts

ContentFilter = Union[Callable, Sequence[str]]
StoreItemType = Any
StoreItemPair = Tuple[str, StoreItemType]
StoreItemMetaData = dict


@dataclass
class SourceInfo:
    """Container for a content information"""

    name_to_filename: dict
    names: List[str]
    metadata: Sequence[Dict[str, StoreItemMetaData]]

    def get_names(self, *, name_filter: ContentFilter = None, name_pattern: str = None) -> List[str]:
        return [x for x in self.names if filename_satisfied_by(x, name_filter, name_pattern)]

    def get_metadata(self, *, name_filter: ContentFilter = None, name_pattern: str = None) -> List[str]:
        return [m for m in self.metadata if filename_satisfied_by(m['filename'], name_filter, name_pattern)]

    def to_stored_name(self, name: str) -> str:
        return self.name_to_filename.get(name, None)


class ISource(AbstractContextManager):
    @abc.abstractmethod
    def namelist(self, *, pattern: str) -> List[str]:
        ...

    @abc.abstractmethod
    def read(self, filename: str, as_binary: bool = False) -> Optional[Any]:
        ...

    @abc.abstractmethod
    def exists(self, filename: str) -> bool:
        ...

    def get_info(self, opts: TextReaderOpts) -> SourceInfo:
        filenames = self.namelist(pattern=opts.filename_pattern)
        basenames = strip_paths(filenames)
        return SourceInfo(
            name_to_filename={strip_paths(name): filename for name, filename in zip(basenames, filenames)},
            names=basenames,
            metadata=extract_filenames_metadata(filenames=basenames, filename_fields=opts.filename_fields),
        )
