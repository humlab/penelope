from typing import Iterable, Mapping, Tuple

import more_itertools
import pandas as pd
from penelope.type_alias import FilenameTokensTuples
from penelope.utility import getLogger, strip_path_and_extension
from tqdm.auto import tqdm

from .co_occurrence import corpus_co_occurrence
from .interface import ContextOpts, CoOccurrenceError

logger = getLogger('penelope')


def partitioned_corpus_co_occurrence(
    stream: FilenameTokensTuples,
    *,
    document_index: pd.DataFrame,
    token2id: Mapping[str, int],
    context_opts: ContextOpts,
    global_threshold_count: int,
    partition_column: str = 'year',
) -> pd.DataFrame:

    # FIXME: #27 Adding document_index and token2id as parameters causes index updates not to be reflected
    if token2id is None:
        if not hasattr(stream, 'token2id'):
            raise CoOccurrenceError("expected `token2id` found None")
        token2id = stream.token2id

    if document_index is None:
        if not hasattr(stream, 'document_index'):
            raise CoOccurrenceError("expected document index found None")
        document_index = stream.document_index

    if partition_column not in document_index.columns:
        raise CoOccurrenceError(f"expected `{partition_column}` not found in document index")

    if not isinstance(global_threshold_count, int) or global_threshold_count < 1:
        global_threshold_count = 1

    def get_bucket_key(item: Tuple[str, Iterable[str]]) -> int:

        if not isinstance(item, tuple):
            raise CoOccurrenceError(f"expected stream of (name,tokens) tuples found {type(item)}")

        if not isinstance(item[0], str):
            raise CoOccurrenceError(f"expected filename (str) ound {type(item[0])}")

        document_name = strip_path_and_extension(item[0])
        return int(document_index.loc[document_name][partition_column])

    total_results = []
    key_streams = more_itertools.bucket(stream, key=get_bucket_key, validator=None)
    keys = sorted(list(key_streams))

    metadata = []
    for i, key in tqdm(enumerate(keys), position=0, leave=True):

        key_stream: FilenameTokensTuples = key_streams[key]
        keyed_document_index = document_index[document_index[partition_column] == key]

        metadata.append(
            {
                'document_id': i,
                'filename': f'{partition_column}{key}.txt',
                'document_name': f'{partition_column}{key}',
                partition_column: key,
                'n_docs': len(keyed_document_index),
            }
        )

        logger.info(f'Processing {key}...')

        co_occurrence = corpus_co_occurrence(
            key_stream,
            document_index=keyed_document_index,
            token2id=token2id,
            context_opts=context_opts,
            threshold_count=1,
        )

        co_occurrence[partition_column] = key

        # FIXME! #26  n_raw_tokens  n_tokens in document_index ARE EMPRY
        total_results.append(co_occurrence)

    logger.info('Concatenating results...')

    co_occurrences = pd.concat(total_results, ignore_index=True)

    # FIXME: #13 Count threshold value should specify min inclusion value
    logger.info('Filtering results...')
    co_occurrences = _filter_co_coccurrences_by_global_threshold(co_occurrences, global_threshold_count)
    logger.info('Done computing co-occurrences')
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
