from __future__ import annotations

import abc
import csv
import zipfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Set, Union

from penelope.utility import FilenameFieldSpecs

if TYPE_CHECKING:
    from ..document_index import DocumentIndex

# pylint: disable=too-many-instance-attributes

TextSource = Union[str, zipfile.ZipFile, List, Any]

FilenameOrCallableOrSequenceFilter = Union[Callable, Sequence[str]]

GLOBAL_TF_THRESHOLD_MASK_TOKEN: str = "__low-tf__"


@dataclass
class TextReaderOpts:
    filename_pattern: str = field(default="*.txt")
    filename_filter: Optional[FilenameOrCallableOrSequenceFilter] = None
    filename_fields: Optional[FilenameFieldSpecs] = None
    index_field: Optional[str] = None
    as_binary: Optional[bool] = False
    sep: Optional[str] = field(default='\t')
    quoting: Optional[int] = csv.QUOTE_NONE
    n_processes: int = 1
    n_chunksize: int = 2

    @property
    def props(self) -> dict:
        return dict(
            filename_pattern=self.filename_pattern,
            filename_filter=self.filename_filter,
            filename_fields=self.filename_fields,
            index_field=self.index_field,
            as_binary=self.as_binary,
            sep=self.sep,
            quoting=self.quoting,
            n_processes=self.n_processes,
            n_chunksize=self.n_chunksize,
        )

    def copy(self, **kwargs) -> TextReaderOpts:
        return TextReaderOpts(**{**self.props, **kwargs})

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


PhraseSubstitutions = Union[Dict[str, List[str]], List[List[str]]]


@dataclass
class ExtractTaggedTokensOpts:

    lemmatize: Optional[bool] = None

    text_column: str = 'token'
    lemma_column: str = 'baseform'
    pos_column: str = 'pos'

    target_override: Optional[str] = None

    """ These PoS define the tokens of interest """
    pos_includes: str = ''

    """ These PoS are always removed """
    pos_excludes: str = ''

    """ The PoS define tokens that are replaced with a dummy marker `*` """
    pos_paddings: Optional[str] = ''
    pos_replace_marker: str = '*'

    passthrough_tokens: List[str] = field(default_factory=list)
    append_pos: bool = False

    phrases: Optional[PhraseSubstitutions] = None

    """Tokens that will be filtered out"""
    block_tokens: List[str] = field(default_factory=list)

    # block_chars: str = ""

    """Global term frequency threshold"""
    global_tf_threshold: int = 1

    """Global term frequency threshold"""
    global_tf_threshold_mask: bool = False

    @property
    def target_column(self) -> str:
        if self.target_override:
            return self.target_override
        return self.lemma_column if self.lemmatize else self.text_column

    def get_pos_includes(self) -> Set[str]:
        return set(self.pos_includes.strip('|').split('|')) if self.pos_includes else set()

    def get_pos_excludes(self) -> Set[str]:
        return set(self.pos_excludes.strip('|').split('|')) if self.pos_excludes is not None else set()

    def get_pos_paddings(self) -> Set[str]:
        return (
            set(x for x in self.pos_paddings.strip('|').split('|') if x != '')
            if self.pos_paddings is not None
            else set()
        )

    def get_passthrough_tokens(self) -> Set[str]:
        if self.passthrough_tokens is None:
            return set()
        return set(self.passthrough_tokens)

    def get_block_tokens(self) -> Set[str]:
        if self.block_tokens is None:
            return set()
        return set(self.block_tokens)

    def clear_tf_threshold(self) -> ExtractTaggedTokensOpts:
        self.global_tf_threshold = 1
        return self

    @property
    def props(self):
        return dict(
            lemmatize=self.lemmatize,
            target_override=self.target_override,
            pos_includes=self.pos_includes,
            pos_excludes=self.pos_excludes,
            pos_paddings=self.pos_paddings,
            pos_replace_marker=self.pos_replace_marker,
            passthrough_tokens=list(self.passthrough_tokens or []),
            block_tokens=list(self.block_tokens or []),
            append_pos=self.append_pos,
            phrases=None if self.phrases is None else list(self.phrases),
            text_column=self.text_column,
            lemma_column=self.lemma_column,
            pos_column=self.pos_column,
        )


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
    def document_index(self) -> DocumentIndex:
        return None

    @abc.abstractmethod
    def __next__(self):
        'Return the next item from the iterator. When exhausted, raise StopIteration'
        raise StopIteration

    @abc.abstractmethod
    def __iter__(self) -> "ICorpusReader":
        return self
