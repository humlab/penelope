import abc
import csv
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd
from penelope.utility import FilenameFieldSpecs, pandas_utils

TextSource = Union[str, zipfile.ZipFile, List, Any]

FilenameOrCallableOrSequenceFilter = Union[Callable, Sequence[str]]


@dataclass
class TextReaderOpts:
    filename_pattern: str = field(default="*.txt")
    filename_filter: Optional[FilenameOrCallableOrSequenceFilter] = None
    filename_fields: Optional[FilenameFieldSpecs] = None
    index_field: Optional[str] = None
    as_binary: Optional[bool] = False
    sep: Optional[str] = field(default='\t')
    quoting: Optional[int] = csv.QUOTE_NONE

    @property
    def props(self):
        return dict(
            filename_pattern=self.filename_pattern,
            filename_filter=self.filename_filter,
            filename_fields=self.filename_fields,
            index_field=self.index_field,
            as_binary=self.as_binary,
            sep=self.sep,
            quoting=self.quoting,
        )

    def copy(self, **kwargs):
        return TextReaderOpts(**{**self.props, **kwargs})


@dataclass
class ExtractTaggedTokensOpts:

    # FIXME: Removed optional, change default to False if optional
    lemmatize: bool  # = True

    target_override: str = None

    pos_includes: str = ''
    # FIXME: Changed default, investigate use, force in Sparv extracts
    pos_excludes: str = ''  # "|MAD|MID|PAD|"

    # FIXME: Implement in spaCy extract
    passthrough_tokens: List[str] = field(default_factory=list)
    append_pos: bool = False

    def get_pos_includes(self):
        return self.pos_includes.strip('|').split('|') if self.pos_includes else None

    def get_pos_excludes(self):
        return self.pos_excludes.strip('|').split('|') if self.pos_excludes is not None else None

    def get_passthrough_tokens(self) -> Set[str]:
        if self.passthrough_tokens is None:
            return set()
        return set(self.passthrough_tokens)

    @property
    def props(self):
        return dict(
            pos_includes=self.pos_includes,
            pos_excludes=self.pos_excludes,
            passthrough_tokens=(self.passthrough_tokens or []),
            lemmatize=self.lemmatize,
            append_pos=self.append_pos,
        )


class TaggedTokensFilterOpts:
    """Used for filtering tagged data that are stored as Pandas data frames.
    A simple key-value filter that returns a mask set to True for items that fulfills all criterias"""

    def __init__(self, **kwargs):
        super().__setattr__('data', kwargs or dict())

    def __getitem__(self, key: int):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __setattr__(self, k, v):
        self.data[k] = v

    def __getattr__(self, k):
        try:
            return self.data[k]
        except KeyError:
            return None

    @property
    def props(self) -> Dict:
        return self.data

    def mask(self, doc: pd.DataFrame) -> np.ndarray:

        return pandas_utils.create_mask(doc, self.data)

    def apply(self, doc: pd.DataFrame) -> pd.DataFrame:
        if len(self.hot_attributes(doc)) == 0:
            return doc
        return doc[self.mask(doc)]

    def hot_attributes(self, doc: pd.DataFrame) -> List[str]:
        """Returns attributes that __might__ filter tagged frame"""
        return [
            (attr_name, attr_value)
            for attr_name, attr_value in self.data.items()
            if attr_name in doc.columns and attr_value is not None
        ]


class ICorpusReader(abc.ABC):
    @property
    @abc.abstractproperty
    def filenames(self) -> List[str]:
        return None

    @property
    @abc.abstractproperty
    def metadata(self) -> List[Dict[str, Any]]:
        return None

    @property
    def document_index(self) -> pd.DataFrame:
        return None

    @abc.abstractmethod
    def __next__(self):
        'Return the next item from the iterator. When exhausted, raise StopIteration'
        raise StopIteration

    @abc.abstractmethod
    def __iter__(self) -> "ICorpusReader":
        return self
