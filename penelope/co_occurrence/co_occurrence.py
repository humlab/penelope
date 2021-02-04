from __future__ import annotations

import collections
import itertools
from typing import TYPE_CHECKING, Iterable, List, Mapping

import pandas as pd
from penelope.corpus import CorpusVectorizer, VectorizedCorpus
from penelope.type_alias import FilenameTokensTuples

from .convert import to_dataframe
from .interface import ContextOpts, CoOccurrenceError
from .windows_corpus import WindowsCorpus

if TYPE_CHECKING:
    from penelope.pipeline.interfaces import PipelinePayload


def tokens_to_windows(*, tokens: Iterable[str], context_opts: ContextOpts, padding: str = '*') -> Iterable[List[str]]:
    """Yields sliding windows of size `2 * context_opts.context_width + 1` for `tokens`


    If `context_opts.concept` is specified then **only windows centered** on any of the
    specified  token stored in `concept` are yielded. All other windows are skipped.

    `context_opts.context_width` is the the number of tokens to either side of the docus word, i.e.
    the total size of the window is (n_window + 1 + n_window).


    Uses the "deck" `collection.deque` with a fixed length (appends exceeding `maxlen` deletes oldest entry)
    The yelded windows are all equal-sized with the focus `*`-padded at the beginning and end
    of the token sequence.

    Parameters
    ----------
    tokens : Iterable[str]
        The sequence of tokens to be windowed
    context_opts: ContextOpts
        context_width : int
            The number of tokens to either side of the token in focus.
        concept : Sequence[str]
            The token(s) in focus.
        ignore_concept: bool
            If to then filter ut the foxus word.

    Yields
    -------
    Iterable[List[str]]
        The sequence of windows
    """

    n_window = 2 * context_opts.context_width + 1

    padded_tokens = itertools.chain(
        [padding] * context_opts.context_width, tokens, [padding] * context_opts.context_width
    )

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


def corpus_to_windows(*, stream: FilenameTokensTuples, context_opts: ContextOpts, pad: str = "*") -> Iterable[List]:

    win_iter = (
        [filename, i, window]
        for filename, tokens in stream
        for i, window in enumerate(tokens_to_windows(tokens=tokens, context_opts=context_opts, padding=pad))
    )
    return win_iter


def corpus_co_occurrence(
    stream: FilenameTokensTuples,
    *,
    payload: PipelinePayload,
    context_opts: ContextOpts,
    threshold_count: int = 1,
) -> pd.DataFrame:
    """Computes a concept co-occurrence dataframe for given arguments

    Parameters
    ----------
    stream : FilenameTokensTuples
        If stream from TokenizedCorpus: Tokenized stream of (filename, tokens)
        If stream from pipeline: sequence of document payloads
    context_opts : ContextOpts
        The co-occurrence opts (context width, optionally concept opts)
    threshold_count : int, optional
        Co-occurrence count filter threshold to use, by default 1

    Returns
    -------
    [type]
        [description]
    """
    if payload.document_index is None:
        raise CoOccurrenceError("expected document index found None")

    if payload.token2id is None:
        raise CoOccurrenceError("expected `token2id` found None")

    windowed_corpus = to_vectorized_windows_corpus(stream=stream, token2id=payload.token2id, context_opts=context_opts)

    co_occurrence_matrix = windowed_corpus.co_occurrence_matrix()

    co_occurrences: pd.DataFrame = to_dataframe(
        co_occurrence_matrix,
        id2token=windowed_corpus.id2token,
        document_index=payload.document_index,
        threshold_count=threshold_count,
    )

    return co_occurrences


def to_vectorized_windows_corpus(
    *,
    stream: FilenameTokensTuples,
    token2id: Mapping[str, int],
    context_opts: ContextOpts,
) -> VectorizedCorpus:
    windows = corpus_to_windows(stream=stream, context_opts=context_opts, pad='*')
    windows_corpus = WindowsCorpus(windows=windows, vocabulary=token2id)
    corpus: VectorizedCorpus = CorpusVectorizer().fit_transform(
        windows_corpus, vocabulary=token2id, already_tokenized=True
    )
    return corpus
