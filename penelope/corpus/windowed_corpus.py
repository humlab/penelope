import collections
import itertools
from typing import Iterable, Set

from penelope.corpus.tokenized_corpus import TokenizedCorpus


def concept_windows(tokens: Iterable[str], concept: Set[str], n_tokens: int, padding='*'):
    """Yields a sequence of windows centered on any of the concept's token stored in `concept`.
    `n_window` is the the number of tokens to either side of the docus word, i.e.
    the total size of the window is (n_window + 1 + n_window).

    Uses the "deck" `collection.deque` with a fixed length (appends exceeding `maxlen` deletes oldest entry)
    The yelded windows are all equal-sized with the focus `*`-padded at the beginning and end
    of the token sequence.

    Parameters
    ----------
    tokens : Iterable[str]
        The sequence of tokens to be windowed
    concept : Sequence[str]
        A set of concept words.
    n_tokens : int
        The number of tokens to either side of the concept token in focus.

    Returns
    -------
    Sequence[str]
        The window

    Yields
    -------
    [type]
        [description]
    """

    n_window = 2 * n_tokens + 1

    _tokens = itertools.chain([padding] * n_tokens, tokens, [padding] * n_tokens)
    # _tokens = iter(_tokens)

    # Fill a full window minus 1
    window = collections.deque((next(_tokens, None) for _ in range(0, n_window - 1)), maxlen=n_window)
    for token in _tokens:
        window.append(token)
        if window[n_tokens] in concept:
            yield list(window)


def corpus_concept_windows(corpus: TokenizedCorpus, concept: Set, n_tokens: int, pad: str = "*"):

    for filename, tokens in corpus:
        for i, window in enumerate(concept_windows(tokens, concept, n_tokens, padding=pad)):
            yield [filename, i, window]
