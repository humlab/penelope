from collections import defaultdict
from typing import Any, Dict, Iterator, List, Mapping, Tuple

import pandas as pd
from penelope.corpus.interfaces import ITokenizedCorpus

WindowsStream = Iterator[Tuple[str, int, Iterator[str]]]


class WindowsCorpus(ITokenizedCorpus):
    """Aggregates statistics while iterating a stream of token windows.
    Each window is a tuple (filename: str, id: int, tokens: Iterator[str])
    """

    def __init__(self, windows: WindowsStream, vocabulary: Mapping[str, int] = None):
        """[summary]

        Parameters
        ----------
        windows : WindowsStream
            Stream of windows to iterate over
        """
        self.statistics = defaultdict(lambda: {'n_windows': 0, 'n_tokens': 0})
        self.windows = iter(windows)
        self._documents: pd.DataFrame = None
        self._metadata = []
        self._vocabulary = vocabulary

    def __iter__(self):
        return self

    def __next__(self):
        try:
            filename, _, tokens = next(self.windows)
            _stats = self.statistics[filename]
            _stats['n_windows'] = _stats['n_windows'] + 1
            _stats['n_tokens'] = _stats['n_tokens'] + len(tokens)
            return (filename, tokens)
        except StopIteration:
            self._metadata = [{'filename': k, **v} for k, v in dict(self.statistics).items()]
            self._documents = pd.DataFrame(self._metadata)
            raise

    @property
    def documents(self) -> pd.DataFrame:
        return self._documents

    @property
    def terms(self) -> Iterator[Iterator[str]]:
        return (tokens for _, tokens in self)

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        return self._metadata

    @property
    def filenames(self) -> List[str]:
        return [d['filename'] for d in self._metadata]

    @property
    def vocabulary(self) -> List[str]:
        return self._vocabulary
