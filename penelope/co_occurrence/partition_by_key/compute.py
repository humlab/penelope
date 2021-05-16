from typing import Iterable, Tuple

import more_itertools
import pandas as pd
from penelope.corpus import DocumentIndex, DocumentIndexHelper, Token2Id, TokensTransformOpts
from penelope.type_alias import FilenameTokensTuples
from penelope.utility import strip_path_and_extension
from tqdm.auto import tqdm

from ..interface import ContextOpts, CoOccurrenceComputeResult, CoOccurrenceError
from .convert import to_dataframe, to_vectorized_windows_corpus

# pylint: disable=ungrouped-imports


# FIXME: #94 Enable partition by alternative keys (apart from year)
def compute_corpus_co_occurrence(
    stream: FilenameTokensTuples,
    *,
    token2id: Token2Id,
    document_index: DocumentIndex,
    context_opts: ContextOpts,
    transform_opts: TokensTransformOpts,
    partition_key: str,
    global_threshold_count: int,
    ignore_pad: str = None,
) -> CoOccurrenceComputeResult:

    if token2id is None:
        # if not hasattr(stream, 'token2id'):
        raise CoOccurrenceError("expected `token2id` found None")
        # payload.token2id = stream.token2id

    if not isinstance(token2id, Token2Id):
        # if not hasattr(stream, 'token2id'):
        raise TypeError("`token2id` no instance of Token2Id")
        # payload.token2id = stream.token2id

    if document_index is None:
        # if not hasattr(stream, 'document_index'):
        raise CoOccurrenceError("expected document index found None")
        # payload._document_index = stream.document_index

    if partition_key not in document_index.columns:
        raise CoOccurrenceError(f"expected `{partition_key}` not found in document index")

    if not isinstance(global_threshold_count, int) or global_threshold_count < 1:
        global_threshold_count = 1

    # FIXME #101 performance: get_bucket_key resolves entire pipeline (inefficient)
    def get_bucket_key(item: Tuple[str, Iterable[str]]) -> int:

        if not isinstance(item, tuple):
            raise CoOccurrenceError(f"expected stream of (name,tokens) tuples found {type(item)}")

        if not isinstance(item[0], str):
            raise CoOccurrenceError(f"expected filename (str) ound {type(item[0])}")

        document_name = strip_path_and_extension(item[0])
        return document_index.loc[document_name][partition_key]

    total_results = []
    key_streams = more_itertools.bucket(stream, key=get_bucket_key, validator=None)
    keys = sorted(list(key_streams))

    # metadata: List[dict] = []
    # FIXME #106 Skip buckets if document-wise compute (i.e. key == document_name)
    # FIXME #107 Parallelize if document-wise compute (i.e. key == document_name)
    for _, key in tqdm(enumerate(keys), desc="Processing partitions", position=0, leave=True):

        key_stream: FilenameTokensTuples = key_streams[key]

        # FIXME #90 Co-occurrence: Enable document based co-occurrence computation
        co_occurrence: pd.DataFrame = compute_co_occurrence(
            key_stream,
            token2id=token2id,
            document_index=document_index,
            context_opts=context_opts,
            threshold_count=1,
            ignore_pad=ignore_pad,
            transform_opts=transform_opts,
        )

        co_occurrence[partition_key] = key

        total_results.append(co_occurrence)

    co_occurrences: pd.DataFrame = pd.concat(total_results, ignore_index=True)

    co_document_index: DocumentIndex = (
        document_index
        if partition_key in ('document_name', 'document_id')
        else (
            DocumentIndexHelper(document_index).group_by_column(column_name=partition_key, index_values=keys)
        ).document_index
    )

    co_occurrences = _filter_co_coccurrences_by_global_threshold(co_occurrences, global_threshold_count)

    return CoOccurrenceComputeResult(co_occurrences=co_occurrences, document_index=co_document_index)


def compute_co_occurrence(
    stream: FilenameTokensTuples,
    *,
    token2id: Token2Id,
    document_index: DocumentIndex,
    context_opts: ContextOpts,
    threshold_count: int = 1,
    ignore_pad: str = None,
    transform_opts: TokensTransformOpts = None,
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
    pd.DataFrame
        Co-occurrence matrix represented via a data frame
    """
    if document_index is None:
        raise CoOccurrenceError("expected document index found None")

    if token2id is None:
        raise CoOccurrenceError("expected `token2id` found None")

    # FIXME: #91 Co-occurrence: Add counter for number of windows a word occurs in
    # FIXME: #92 Co-occurrence: Add counter for number of windows a word-pair occurs in
    windowed_corpus = to_vectorized_windows_corpus(stream=stream, token2id=token2id, context_opts=context_opts)

    co_occurrence_matrix = windowed_corpus.co_occurrence_matrix()

    co_occurrences: pd.DataFrame = to_dataframe(
        co_occurrence_matrix,
        id2token=windowed_corpus.id2token,
        document_index=document_index,
        threshold_count=threshold_count,
        ignore_pad=ignore_pad,
        transform_opts=transform_opts,
    )

    return co_occurrences


def _filter_co_coccurrences_by_global_threshold(co_occurrences: pd.DataFrame, threshold: int) -> pd.DataFrame:
    if len(co_occurrences) == 0:
        return co_occurrences
    if threshold is None or threshold <= 1:
        return co_occurrences
    filtered_co_occurrences = co_occurrences[
        co_occurrences.groupby(["w1", "w2"])['value'].transform('sum') >= threshold
    ]
    return filtered_co_occurrences
