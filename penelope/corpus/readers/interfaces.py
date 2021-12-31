from __future__ import annotations

import abc
import csv
import zipfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Set, Union

from penelope.utility import FilenameFieldSpecs, PropertyValueMaskingOpts, PropsMixIn

if TYPE_CHECKING:
    from ..document_index import DocumentIndex

# pylint: disable=too-many-instance-attributes

TextSource = Union[str, zipfile.ZipFile, List, Any]

FilenameFilterSpec = Union[Callable, Sequence[str]]

GLOBAL_TF_THRESHOLD_MASK_TOKEN: str = "__low-tf__"


@dataclass
class TextReaderOpts(PropsMixIn["TextReaderOpts"]):
    filename_pattern: str = field(default="*.txt")
    filename_filter: Optional[FilenameFilterSpec] = None
    filename_fields: Optional[FilenameFieldSpecs] = None
    index_field: Optional[str] = None
    as_binary: Optional[bool] = False
    sep: Optional[str] = field(default='\t')
    quoting: Optional[int] = csv.QUOTE_NONE
    n_processes: int = 1
    n_chunksize: int = 2
    dehyphen_expr: str = field(default=r"\b(\w+)[-¬]\s*\r?\n\s*(\w+)\s*\b")

    def copy(self, **kwargs) -> TextReaderOpts:
        return TextReaderOpts(**{**self.props, **kwargs})


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

    """ These PoS define tokens that are replaced with a dummy marker `*` """
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

    filter_opts: Union[PropertyValueMaskingOpts, dict] = field(default_factory=PropertyValueMaskingOpts)

    def __post_init__(self):
        self.filter_opts = (
            PropertyValueMaskingOpts(**self.filter_opts)
            if isinstance(self.filter_opts, dict)
            else self.filter_opts or PropertyValueMaskingOpts()
        )

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
            filter_opts=self.filter_opts.props,
        )

    def ingest(self, opts: dict) -> ExtractTaggedTokensOpts:
        self.pos_includes = opts.get('pos_includes', self.pos_includes)
        self.pos_paddings = opts.get('pos_paddings', self.pos_paddings)
        self.pos_excludes = opts.get('pos_excludes', self.pos_excludes)
        self.lemmatize = opts.get('lemmatize', self.lemmatize)
        self.phrases = opts.get('phrases', self.phrases)
        self.append_pos = opts.get('append_pos', self.append_pos)
        self.global_tf_threshold = opts.get('tf_threshold', self.global_tf_threshold)
        self.global_tf_threshold_mask = opts.get('tf_threshold_mask', self.global_tf_threshold_mask)
        return self

    def set_numeric_names(self) -> ExtractTaggedTokensOpts:
        self.pos_column = "pos_id"
        self.lemma_column = "lemma_id"
        self.text_column = "token_id"
        return self

    @property
    def has_effect(self) -> bool:
        if self.pos_includes or self.pos_paddings or self.pos_excludes:
            return True
        if self.phrases:
            return True
        if self.append_pos:
            return True
        if self.global_tf_threshold > 1:
            return True
        return False

    @property
    def of_no_effect(self) -> bool:
        return not self.has_effect


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
