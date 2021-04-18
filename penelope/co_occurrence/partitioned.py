from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Tuple

import more_itertools
import pandas as pd
from penelope.corpus import DocumentIndex, DocumentIndexHelper
from penelope.utility import getLogger, strip_path_and_extension
from tqdm.auto import tqdm

from .co_occurrence import corpus_co_occurrence
from .interface import ContextOpts, CoOccurrenceError

# pylint: disable=ungrouped-imports

if TYPE_CHECKING:
    from penelope.pipeline import PipelinePayload
    from penelope.type_alias import FilenameTokensTuples


logger = getLogger('penelope')


@dataclass
class ComputeResult:
    co_occurrences: pd.DataFrame = None
    document_index: DocumentIndex = None


def partitioned_corpus_co_occurrence(
    stream: FilenameTokensTuples,
    *,
    payload: PipelinePayload,
    context_opts: ContextOpts,
    global_threshold_count: int,
    partition_column: str = 'year',
    ignore_pad: str = None,
) -> ComputeResult:

    # FIXME: #27 Adding document_index and token2id as parameters causes index updates not to be reflected
    if payload.token2id is None:
        # if not hasattr(stream, 'token2id'):
        raise CoOccurrenceError("expected `token2id` found None")
        # payload.token2id = stream.token2id

    if payload.document_index is None:
        # if not hasattr(stream, 'document_index'):
        raise CoOccurrenceError("expected document index found None")
        # payload._document_index = stream.document_index

    if partition_column not in payload.document_index.columns:
        raise CoOccurrenceError(f"expected `{partition_column}` not found in document index")

    if not isinstance(global_threshold_count, int) or global_threshold_count < 1:
        global_threshold_count = 1

    def get_bucket_key(item: Tuple[str, Iterable[str]]) -> int:

        if not isinstance(item, tuple):
            raise CoOccurrenceError(f"expected stream of (name,tokens) tuples found {type(item)}")

        if not isinstance(item[0], str):
            raise CoOccurrenceError(f"expected filename (str) ound {type(item[0])}")

        document_name = strip_path_and_extension(item[0])
        return int(payload.document_index.loc[document_name][partition_column])

    total_results = []
    key_streams = more_itertools.bucket(stream, key=get_bucket_key, validator=None)
    keys = sorted(list(key_streams))

    # metadata: List[dict] = []
    for _, key in tqdm(enumerate(keys), desc="Processing partitions", position=0, leave=True):

        key_stream: FilenameTokensTuples = key_streams[key]

        # keyed_document_index: DocumentIndex= payload.document_index[payload.document_index[partition_column] == key]
        # metadata.append(_group_metadata(keyed_document_index, i, partition_column, key))

        co_occurrence = corpus_co_occurrence(
            key_stream,
            payload=payload,
            context_opts=context_opts,
            threshold_count=1,
            ignore_pad=ignore_pad,
        )

        co_occurrence[partition_column] = key

        # FIXME! #26  n_raw_tokens  n_tokens in document_index ARE EMPRY
        total_results.append(co_occurrence)

    co_occurrences = pd.concat(total_results, ignore_index=True)

    # metadata_document_index: DocumentIndex = DocumentIndexHelper.from_metadata(metadata).document_index

    index: DocumentIndex = (
        DocumentIndexHelper(payload.document_index).group_by_column(column_name=partition_column, index_values=keys)
    ).document_index

    # FIXME: #13 Count threshold value should specify min inclusion value
    co_occurrences = _filter_co_coccurrences_by_global_threshold(co_occurrences, global_threshold_count)

    return ComputeResult(co_occurrences=co_occurrences, document_index=index)


# def _group_metadata(keyed_document_index: DocumentIndex, i: int, column_name: str, value: Union[int,str]) -> dict:
#     return {
#         **{
#             'document_id': i,
#             'filename': f'{column_name}_{value}.txt',
#             'document_name': f'{column_name}_{value}',
#             'category': value,
#             'n_docs': len(keyed_document_index),
#             'year': value if column_name == 'year' else keyed_document_index.year.min()
#         },
#         **{
#             count_column: keyed_document_index[count_column].sum().astype(np.int64)
#             for count_column in DOCUMENT_INDEX_COUNT_COLUMNS
#             if count_column in keyed_document_index.columns
#         },
#     }


def _filter_co_coccurrences_by_global_threshold(co_occurrences: pd.DataFrame, threshold: int) -> pd.DataFrame:
    if len(co_occurrences) == 0:
        return co_occurrences
    if threshold is None or threshold <= 1:
        return co_occurrences
    filtered_co_occurrences = co_occurrences[
        co_occurrences.groupby(["w1", "w2"])['value'].transform('sum') >= threshold
    ]
    return filtered_co_occurrences
