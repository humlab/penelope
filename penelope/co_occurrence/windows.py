import collections
import itertools
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Tuple

from penelope.corpus import DocumentIndex, ITokenizedCorpus, metadata_to_document_index
from penelope.type_alias import Token, WindowsStream
from penelope.utility import deprecated

from .interface import ContextOpts


class WindowsCorpus(ITokenizedCorpus):
    """Aggregates statistics while iterating the document stream with sliding windows.
    Each window is a triple (filename: str, id: int, tokens: Iterator[str])
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


@deprecated
def tokens_to_windows_(*, tokens: Iterable[Token], context_opts: ContextOpts) -> Iterable[Iterable[Token]]:
    """Yields sliding windows of size `2 * context_opts.context_width + 1` for `tokens`


    If `context_opts.concept` is specified then **only windows centered** on any of the
    specified  token stored in `concept` are yielded. All other windows are skipped.

    `context_opts.context_width` is the the number of tokens to either side of the focus word, i.e.
    the total size of the window is (n_window + 1 + n_window).

    If `context_opts.ignore_padding` is true, then the padding token is removed from returned windows.
    The returned has varying sizes in such a case.

    Uses the "deck" `collection.deque` with a fixed length (appends exceeding `maxlen` deletes oldest entry)
    The yelded windows are all equal-sized with the focus `*`-padded at the beginning and end
    of the token sequence.

    Parameters
    ----------
    tokens : Iterable[Token]
        The sequence of tokens to be windowed
    context_opts: ContextOpts (width, concept, etc)

    Yields
    -------
    Iterable[Tuple[List[str], bool]]
        The sequence of windows and flag that is true if window is a concept context
    """

    pad: Token = context_opts.pad

    n_window = 2 * context_opts.context_width + 1

    padded_tokens = itertools.chain([pad] * context_opts.context_width, tokens, [pad] * context_opts.context_width)

    window = collections.deque((next(padded_tokens, None) for _ in range(0, n_window - 1)), maxlen=n_window)

    if not context_opts.concept:

        """ Co-occurrence without concept words """

        for token in padded_tokens:

            window.append(token)

            next_window = [w for w in window if w != pad] if context_opts.ignore_padding else list(window)

            if len(next_window) >= context_opts.min_window_size:
                yield next_window

    else:

        """ Co-occurrence with concept words """

        for token in padded_tokens:

            window.append(token)

            if window[context_opts.context_width] in context_opts.concept:

                """ Context window """
                concept_window = list(window)

                if context_opts.ignore_concept:
                    _ = concept_window.pop(context_opts.context_width)

                if context_opts.ignore_padding:
                    concept_window = [w for w in concept_window if w != pad]

                if len(concept_window) >= context_opts.min_window_size:
                    yield concept_window


def generate_windows(
    *,
    token_ids: Iterable[int],
    context_width: int,
    pad_id: int,
    ignore_pads: bool = False,
) -> Iterable[Iterable[int]]:
    """Yields sliding windows of size `2 * context_opts.context_width + 1` for `tokens_ids`

    Uses the "deck" `collection.deque` with a fixed length (appends exceeding `maxlen` deletes oldest entry)
    The yelded windows are all equal-sized with the focus `*`-padded at the beginning and end
    of the token sequence.


    Args:
        token_ids (Iterable[int]): The sequence of tokens_ids to be windowed
        context_width (int): (window size - 1) / 2
        pad_id (int): Token used as magic pad for non-target words (and start/end filler).
        ignore_pads (bool, optional): If true, pad tokens are filtered out. Defaults to True.

    Returns:
        Iterable[Iterable[int]]: [description]

    Yields:
        Iterator[Iterable[Iterable[int]]]: [description]
    """

    n_window = 2 * context_width + 1
    padded_tokens = itertools.chain([pad_id] * context_width, token_ids, [pad_id] * context_width)
    window = collections.deque((next(padded_tokens, None) for _ in range(0, n_window - 1)), maxlen=n_window)

    if ignore_pads:
        for token in padded_tokens:
            window.append(token)
            yield [w for w in window if w != pad_id]
    else:
        for token in padded_tokens:
            window.append(token)
            yield list(window)
