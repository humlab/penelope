from collections import Counter, defaultdict
from typing import Any, Dict, Iterator, List, Mapping, Tuple

from penelope.corpus import ITokenizedCorpus, metadata_to_document_index
from penelope.corpus.document_index import DocumentIndex

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
        self._document_index: DocumentIndex = None
        self._metadata = []
        self._vocabulary = vocabulary
        self._token_windows_counter: Counter = Counter()

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[str, List[str]]:
        try:
            # FIXME #91, #92: Add counters!!!
            filename, _, tokens = next(self.windows)
            _stats = self.statistics[filename]
            _stats['n_windows'] = _stats['n_windows'] + 1
            _stats['n_tokens'] = _stats['n_tokens'] + len(tokens)  # Always equal n_windows * window_size!
            for token in tokens:
                self._token_windows_counter[token] += 1
            return (filename, tokens)
        except StopIteration:
            self._metadata = [{'filename': k, **v} for k, v in dict(self.statistics).items()]
            self._document_index = metadata_to_document_index(self._metadata)
            raise

    @property
    def document_index(self) -> DocumentIndex:
        return self._document_index

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

    @property
    def token_window_counts(self) -> Counter:
        """Returns a counter with token's windows count."""
        return self._token_windows_counter
