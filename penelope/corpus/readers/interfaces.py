import abc
import zipfile
from typing import Any, Dict, List, Union

import pandas as pd
from penelope.corpus.readers.text_reader import TextReader

TextSource = Union[str, zipfile.ZipFile, List, TextReader]


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
