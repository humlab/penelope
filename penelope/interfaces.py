import abc
import collections
from typing import Any, Dict, List


class ICorpusReader(collections.abc.Iterator):
    @property
    @abc.abstractmethod
    def metadata(self) -> List[Dict[str, Any]]:
        pass

    @property
    @abc.abstractmethod
    def filenames(self) -> List[str]:
        pass


# class ICorpusTokenizer(collections.abc.Iterable):
#     pass
