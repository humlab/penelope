import collections
import itertools
from typing import Iterable, List, Set

import pandas as pd
from tqdm import tqdm

from penelope.corpus.interfaces import ICorpus, ITokenizedCorpus, PartitionKeys
from penelope.corpus.vectorizer import CorpusVectorizer
from penelope.utility import file_utility

from .term_term_matrix import to_dataframe
from .windows_corpus import WindowsCorpus


def tokens_concept_windows(
    tokens: Iterable[str], concept: Set[str], no_concept: bool, n_context_width: int, padding='*'
):
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
        The token(s) in focus.
    no_concept: bool
        If to then filter ut the foxus word.
    n_context_width : int
        The number of tokens to either side of the token in focus.

    Returns
    -------
    Sequence[str]
        The window

    Yields
    -------
    [type]
        [description]
    """

    n_window = 2 * n_context_width + 1

    _tokens = itertools.chain([padding] * n_context_width, tokens, [padding] * n_context_width)
    # _tokens = iter(_tokens)

    # FIXME: #7 Add test case for --no-concept option
    # Fill a full window minus 1
    window = collections.deque((next(_tokens, None) for _ in range(0, n_window - 1)), maxlen=n_window)
    for token in _tokens:
        window.append(token)
        if window[n_context_width] in concept:
            concept_window = list(window)
            if no_concept:
                _ = concept_window.pop(n_context_width)
            yield concept_window


def corpus_concept_windows(corpus: ICorpus, concept: Set, no_concept: bool, n_context_width: int, pad: str = "*"):

    win_iter = (
        [filename, i, window]
        for filename, tokens in corpus
        for i, window in enumerate(
            tokens_concept_windows(
                tokens=tokens, concept=concept, no_concept=no_concept, n_context_width=n_context_width, padding=pad
            )
        )
    )
    return win_iter


def corpus_concept_co_occurrence(
    corpus: ITokenizedCorpus,
    *,
    concepts: Set[str],
    no_concept: bool,
    n_context_width: int,
    filenames: List[str] = None,
    count_threshold: int = 1,
):
    """Computes a concept co-occurrence dataframe for given arguments

    Parameters
    ----------
    corpus : ITokenizedCorpus
        Tokenized corpus
    concept : Set[str]
        The concept that the defines the context, always the middle word in the window
    no_concept : bool
        If True then filter out concept from result
    n_context_width : int
        Width of left/right context
    filenames : List[str], optional
        Corpus filename subset, by default None
    count_threshold : int, optional
        Co-occurrence count filter threshold to use, by default 1

    Returns
    -------
    [type]
        [description]
    """

    windows = corpus_concept_windows(
        corpus, concept=concepts, no_concept=no_concept, n_context_width=n_context_width, pad='*'
    )

    windows_corpus = WindowsCorpus(windows=windows, vocabulary=corpus.token2id)
    v_corpus = CorpusVectorizer().fit_transform(windows_corpus)

    coo_matrix = v_corpus.co_occurrence_matrix()

    documents = corpus.documents if filenames is None else corpus.documents[corpus.documents.filename.isin(filenames)]

    df_coo = to_dataframe(coo_matrix, id2token=corpus.id2token, documents=documents, min_count=count_threshold)

    return df_coo


def partitioned_corpus_concept_co_occurrence(
    corpus: ITokenizedCorpus,
    *,
    concepts: Set[str],
    no_concept: bool,
    count_threshold: int,
    n_context_width: int,
    partition_keys: PartitionKeys = 'year',
) -> pd.DataFrame:
    """[summary]

    Parameters
    ----------
    corpus : ITokenizedCorpus
        The source corpus
    concept : Set[str]
        The word(s) in focus
    no_concept: bool
        If True then the focus word is filtered out
    count_threshold : int
        Min number of global co-occurrence count to include in the result
    n_context_width : int
        Number of tokens to either side of concept word
    partition_key : str, optional
        Document field to use when partitioning the data, by default 'year'

    Returns
    -------
    pd.DataFrame
        Co-occurrence matrix as a pd.DataFrame
    """

    partitions = corpus.partition_documents(partition_keys)
    df_total = None

    partition_column = partition_keys if isinstance(partition_keys, str) else '_'.join(partition_keys)

    for partition in tqdm(partitions):

        filenames = partitions[partition]

        corpus.reader.apply_filter(filenames)

        df_partition = corpus_concept_co_occurrence(
            corpus, concepts=concepts, no_concept=no_concept, n_context_width=n_context_width, filenames=filenames
        )

        df_partition[partition_column] = partition

        df_total = df_partition if df_total is None else df_total.append(df_partition, ignore_index=True)

    if isinstance(count_threshold, int) and count_threshold > 1:
        # FIXME: #9 Add unit test for `count_theshold` filter
        df_total = df_total[df_total.groupby(["w1", "w2"]).transform('size') > count_threshold]

    return df_total


def compute_and_store(
    corpus: ITokenizedCorpus,
    concepts: List[str],
    no_concept: bool,
    count_threshold: int,
    n_context_width: int,
    partition_keys: List[str],
    target_filename: str,
):
    """Extracts and stores text documents from a Sparv corpus in CSV format

    Parameters
    ----------
    corpus : ITokenizedCorpus
        Corpus

    """
    coo_df = partitioned_corpus_concept_co_occurrence(
        corpus,
        concepts=concepts,
        no_concept=no_concept,
        count_threshold=count_threshold,
        n_context_width=n_context_width,
        partition_keys=partition_keys,
    )

    _store_to_file(target_filename, coo_df)


def _store_to_file(filename: str, df: pd.DataFrame):
    """Store file to disk"""

    if filename.endswith('zip'):
        archive_name = f"{file_utility.strip_path_and_extension(filename)}.csv"
        compression = dict(method='zip', archive_name=archive_name)
    else:
        compression = 'infer'

    df.to_csv(filename, sep='\t', header=True, compression=compression, decimal=',')
