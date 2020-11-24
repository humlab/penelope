import abc
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Union

import pandas as pd
from penelope.utility import PropsMixIn

TextSource = Union[str, zipfile.ZipFile, List, Any]


@dataclass
class ExtractTokensOpts2(PropsMixIn):
    """Spacy document extract options"""

    target: str = "lemma"
    is_alpha: bool = None
    is_space: bool = False
    is_punct: bool = False
    is_digit: bool = None
    is_stop: bool = None
    include_pos: Set[str] = None
    exclude_pos: Set[str] = None


@dataclass
class ExtractTokensOpts:

    pos_includes: str = ''
    pos_excludes: str = "|MAD|MID|PAD|"
    passthrough_tokens: List[str] = field(default_factory=list)
    lemmatize: bool = True
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
    def __iter__(self):
        return self

    @property
    def metadata_lookup(self):
        return {x['filename']: x for x in (self.metadata or [])}
