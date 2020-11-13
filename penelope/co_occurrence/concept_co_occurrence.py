import collections
import itertools
from dataclasses import dataclass, field
from typing import Iterable, List, Set

import pandas as pd
import scipy
from penelope.corpus import VectorizedCorpus
from penelope.corpus.interfaces import ICorpus, ITokenizedCorpus, PartitionKeys
from penelope.corpus.vectorizer import CorpusVectorizer
from penelope.utility import strip_path_and_extension
from tqdm.auto import tqdm

from .term_term_matrix import to_dataframe
from .windows_corpus import WindowsCorpus


@dataclass
class ConceptContextOpts:

    concept: Set[str] = field(default_factory=set)
    ignore_concept: bool = False
    context_width: int = 2

    @property
    def props(self):
        return {
            'concept': list(self.concept),
            'ignore_concept': self.ignore_concept,
            'context_width': self.context_width,
        }


def tokens_concept_windows(tokens: Iterable[str], concept_opts: ConceptContextOpts, padding='*'):
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
    concept_opts: ConceptContextOpts
        concept : Sequence[str]
            The token(s) in focus.
        ignore_concept: bool
            If to then filter ut the foxus word.
        context_width : int
            The number of tokens to either side of the token in focus.

    Yields
    -------
    Iterable[List[str]]
        The sequence of windows
    """

    n_window = 2 * concept_opts.context_width + 1

    _tokens = itertools.chain([padding] * concept_opts.context_width, tokens, [padding] * concept_opts.context_width)
    # _tokens = iter(_tokens)

    # FIXME: #7 Add test case for --no-concept option
    # Fill a full window minus 1
    window = collections.deque((next(_tokens, None) for _ in range(0, n_window - 1)), maxlen=n_window)
    for token in _tokens:
        window.append(token)
        if window[concept_opts.context_width] in concept_opts.concept:
            concept_window = list(window)
            if concept_opts.ignore_concept:
                _ = concept_window.pop(concept_opts.context_width)
            yield concept_window


def corpus_concept_windows(corpus: ICorpus, concept_opts: ConceptContextOpts, pad: str = "*"):

    win_iter = (
        [filename, i, window]
        for filename, tokens in corpus
        for i, window in enumerate(tokens_concept_windows(tokens=tokens, concept_opts=concept_opts, padding=pad))
    )
    return win_iter


def corpus_concept_co_occurrence(
    corpus: ITokenizedCorpus,
    *,
    concept_opts: ConceptContextOpts,
    filenames: List[str] = None,
    threshold_count: int = 1,
):
    """Computes a concept co-occurrence dataframe for given arguments

    Parameters
    ----------
    corpus : ITokenizedCorpus
        Tokenized corpus
    concept_opts : ConceptCoOccurrenceOpts
        The concept definition (concept tokens, context width, concept remove option)
    filenames : List[str], optional
        Corpus filename subset, by default None
    threshold_count : int, optional
        Co-occurrence count filter threshold to use, by default 1

    Returns
    -------
    [type]
        [description]
    """

    windows = corpus_concept_windows(corpus, concept_opts=concept_opts, pad='*')

    windows_corpus = WindowsCorpus(windows=windows, vocabulary=corpus.token2id)
    v_corpus = CorpusVectorizer().fit_transform(windows_corpus)

    coo_matrix = v_corpus.co_occurrence_matrix()

    documents = corpus.documents if filenames is None else corpus.documents[corpus.documents.filename.isin(filenames)]

    df_coo = to_dataframe(coo_matrix, id2token=corpus.id2token, documents=documents, threshold_count=threshold_count)

    return df_coo


def partitioned_corpus_concept_co_occurrence(
    corpus: ITokenizedCorpus,
    *,
    concept_opts: ConceptContextOpts,
    global_threshold_count: int,
    partition_keys: PartitionKeys = 'year',
) -> pd.DataFrame:
    """[summary]

    Parameters
    ----------
    corpus : ITokenizedCorpus
        The source corpus
    concept_opts: ConceptContextOpts
        concept : Set[str]
            The word(s) in focus
        no_concept: bool
            If True then the focus word is filtered out
        n_context_width : int
            Number of tokens to either side of concept word
    global_threshold_count : int
        Min number of global co-occurrence count (i.e. sum over all partitions) to include in result
    partition_key : str, optional
        Document field to use when partitioning the data, by default 'year'

    Returns
    -------
    pd.DataFrame
        Co-occurrence matrix as a pd.DataFrame
    """
    if not isinstance(global_threshold_count, int) or global_threshold_count < 1:
        global_threshold_count = 1

    partitions = corpus.partition_documents(partition_keys)

    df_total = None

    partition_column = partition_keys if isinstance(partition_keys, str) else '_'.join(partition_keys)

    if len(partitions) == 0:
        raise ValueError(f"No partitions found for key {partition_column}")

    for partition in tqdm(partitions):

        filenames = partitions[partition]

        corpus.reader.apply_filter(filenames)

        df_partition = corpus_concept_co_occurrence(
            corpus,
            concept_opts=concept_opts,
            threshold_count=1,  # no threshold for single partition
            filenames=filenames,
        )

        df_partition[partition_column] = partition

        df_total = df_partition if df_total is None else df_total.append(df_partition, ignore_index=True)

    # FIXME: #13 Count threshold value should specify min inclusion value
    df_total = filter_co_coccurrences_by_global_threshold(df_total, global_threshold_count)

    return df_total


def filter_co_coccurrences_by_global_threshold(co_occurrences: pd.DataFrame, threshold: int) -> pd.DataFrame:
    if len(co_occurrences) == 0:
        return co_occurrences
    if threshold is None or threshold <= 1:
        return co_occurrences
    filtered_co_occurrences = co_occurrences[
        co_occurrences.groupby(["w1", "w2"])['value'].transform('sum') >= threshold
    ]
    return filtered_co_occurrences


def store_co_occurrences(filename: str, df: pd.DataFrame):
    """Store co-occurrence result data to CSV-file"""

    if filename.endswith('zip'):
        archive_name = f"{strip_path_and_extension(filename)}.csv"
        compression = dict(method='zip', archive_name=archive_name)
    else:
        compression = 'infer'

    df.to_csv(filename, sep='\t', header=True, compression=compression, decimal=',')


def load_co_occurrences(filename: str) -> pd.DataFrame:
    """Load co-occurrences from CSV-file"""
    # if filename.endswith('zip'):
    #     archive_name = f"{file_utility.strip_path_and_extension(filename)}.csv"
    #     compression = dict(method='zip', archive_name=archive_name)
    # else:
    #     compression = 'infer'
    df = pd.read_csv(filename, sep='\t', header=0, decimal=',', index_col=0)

    return df


def to_vectorized_corpus(co_occurrences: pd.DataFrame, value_column: str) -> VectorizedCorpus:

    # Create new tokens from the co-occurring pairs
    tokens = co_occurrences.apply(lambda x: f'{x["w1"]}/{x["w2"]}', axis=1)

    # Create a vocabulary
    vocabulary = list(sorted([w for w in set(tokens)]))

    # Create token2id mapping
    token2id = {w: i for i, w in enumerate(vocabulary)}
    years = list(sorted(co_occurrences.year.unique()))
    year2index = {year: i for i, year in enumerate(years)}

    df_yearly_weights = pd.DataFrame(
        data={
            'year_index': co_occurrences.year.apply(lambda y: year2index[y]),
            'token_id': tokens.apply(lambda x: token2id[x]),
            'weight': co_occurrences[value_column],
        }
    )

    coo_matrix = scipy.sparse.coo_matrix(
        (df_yearly_weights.weight, (df_yearly_weights.year_index, df_yearly_weights.token_id))
    )

    documents = pd.DataFrame(
        data={'document_id': list(range(0, len(years))), 'filename': [f'{y}.coo' for y in years], 'year': years}
    )

    v_corpus = VectorizedCorpus(coo_matrix, token2id=token2id, documents=documents)

    return v_corpus
