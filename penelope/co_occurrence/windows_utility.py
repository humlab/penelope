import collections
import itertools
from typing import Iterable, List

from penelope.type_alias import FilenameTokensTuples

from .interface import ContextOpts, Token


def tokens_to_windows(*, tokens: Iterable[Token], context_opts: ContextOpts) -> Iterable[List[Token]]:
    """Yields sliding windows of size `2 * context_opts.context_width + 1` for `tokens`


    If `context_opts.concept` is specified then **only windows centered** on any of the
    specified  token stored in `concept` are yielded. All other windows are skipped.

    `context_opts.context_width` is the the number of tokens to either side of the focus word, i.e.
    the total size of the window is (n_window + 1 + n_window).


    Uses the "deck" `collection.deque` with a fixed length (appends exceeding `maxlen` deletes oldest entry)
    The yelded windows are all equal-sized with the focus `*`-padded at the beginning and end
    of the token sequence.

    Parameters
    ----------
    tokens : Iterable[Token]
        The sequence of tokens to be windowed
    context_opts: ContextOpts
        context_width : int
            The number of tokens to either side of the token in focus.
        concept : Sequence[Token]
            The token(s) in focus.
        ignore_concept: bool
            If to then filter ut the focus word.

    Yields
    -------
    Iterable[List[str]]
        The sequence of windows
    """

    pad: Token = context_opts.pad

    n_window = 2 * context_opts.context_width + 1

    padded_tokens = itertools.chain([pad] * context_opts.context_width, tokens, [pad] * context_opts.context_width)

    window = collections.deque((next(padded_tokens, None) for _ in range(0, n_window - 1)), maxlen=n_window)

    if len(context_opts.concept) == 0:

        for token in padded_tokens:
            window.append(token)
            yield list(window)

    else:

        for token in padded_tokens:
            window.append(token)
            if window[context_opts.context_width] in context_opts.concept:
                concept_window = list(window)
                if context_opts.ignore_concept:
                    _ = concept_window.pop(context_opts.context_width)
                yield concept_window


def corpus_to_windows(*, stream: FilenameTokensTuples, context_opts: ContextOpts) -> Iterable[List]:

    win_iter = (
        [filename, i, window]
        for filename, tokens in stream
        for i, window in enumerate(tokens_to_windows(tokens=tokens, context_opts=context_opts))
    )
    return win_iter
