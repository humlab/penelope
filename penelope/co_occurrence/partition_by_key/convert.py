from typing import Mapping, Union

import numpy as np
import pandas as pd
import scipy
from loguru import logger
from penelope.corpus import CorpusVectorizer, DocumentIndex, TokensTransformer, TokensTransformOpts, VectorizedCorpus
from penelope.type_alias import FilenameTokensTuples

from ..interface import ContextOpts, CoOccurrenceError, PartitionKeyNotUniqueKey
from ..windows_corpus import WindowsCorpus
from ..windows_utility import corpus_to_windows

CoOccurrenceDataFrame = pd.DataFrame


def to_vectorized_windows_corpus(
    *,
    stream: FilenameTokensTuples,
    token2id: Mapping[str, int],
    context_opts: ContextOpts,
) -> VectorizedCorpus:
    windows = corpus_to_windows(stream=stream, context_opts=context_opts)
    windows_corpus = WindowsCorpus(windows=windows, vocabulary=token2id)
    corpus: VectorizedCorpus = CorpusVectorizer().fit_transform(
        windows_corpus,
        vocabulary=token2id,
        already_tokenized=True,
    )
    return corpus


def to_vectorized_corpus(
    *,
    co_occurrences: CoOccurrenceDataFrame,
    document_index: DocumentIndex,
    value_key: str,
    partition_key: Union[int, str],
) -> VectorizedCorpus:
    """Creates a DTM corpus from a co-occurrence result set that was partitioned by `partition_column`."""

    if len(document_index[partition_key].unique()) != len(document_index):
        raise PartitionKeyNotUniqueKey()

    # Create new tokens from the co-occurring pairs
    tokens = co_occurrences.apply(lambda x: f'{x["w1"]}/{x["w2"]}', axis=1)

    # Create a vocabulary & token2id mapping
    token2id = {w: i for i, w in enumerate(sorted([w for w in set(tokens)]))}

    # Create a `partition_column` to index mapping (i.e. `partition_column` to document_id)
    partition2index = document_index.set_index(partition_key).document_id.to_dict()

    df_partition_weights = pd.DataFrame(
        data={
            'partition_index': co_occurrences[partition_key].apply(lambda y: partition2index[y]),
            'token_id': tokens.apply(lambda x: token2id[x]),
            'weight': co_occurrences[value_key],
        }
    )
    # Make certain  matrix gets right shape (otherwise empty documents at the end reduces row count)
    shape = (len(partition2index), len(token2id))
    coo_matrix = scipy.sparse.coo_matrix(
        (df_partition_weights.weight, (df_partition_weights.partition_index, df_partition_weights.token_id)),
        shape=shape,
        dtype=np.uint32,
    )

    document_index = document_index.set_index('document_id', drop=False).rename_axis('').sort_index()

    v_corpus = VectorizedCorpus(coo_matrix, token2id=token2id, document_index=document_index)

    return v_corpus


def co_occurrence_term_term_matrix_to_dataframe(
    term_term_matrix: scipy.sparse.spmatrix,
    id2token: Mapping[int, str],
    document_index: DocumentIndex = None,
    threshold_count: int = 1,
    ignore_pad: str = None,
    transform_opts: TokensTransformOpts = None,
) -> CoOccurrenceDataFrame:
    """Converts a TTM to a Pandas DataFrame

    Parameters
    ----------
    term_term_matrix : scipy.sparse.spmatrix
        [description]
    id2token : Id2Token
        [description]
    document_index : DocumentIndex, optional
        [description], by default None
    threshold_count : int, optional
        Min count (`value`) to include in result, by default 1

    Returns
    -------
    CoOccurrenceDataFrame:
        [description]
    """

    if 'n_tokens' not in document_index.columns:
        raise CoOccurrenceError("expected `document_index.n_tokens`, but found no column")

    if 'n_raw_tokens' not in document_index.columns:
        raise CoOccurrenceError("expected `document_index.n_raw_tokens`, but found no column")

    co_occurrences = (
        pd.DataFrame({'w1_id': term_term_matrix.row, 'w2_id': term_term_matrix.col, 'value': term_term_matrix.data})[
            ['w1_id', 'w2_id', 'value']
        ]
        .sort_values(['w1_id', 'w2_id'])
        .reset_index(drop=True)
    )

    if threshold_count > 0:
        co_occurrences = co_occurrences[co_occurrences.value >= threshold_count]

    if document_index is not None:

        co_occurrences['value_n_d'] = co_occurrences.value / float(len(document_index))

        for n_token_count, target_field in [('n_tokens', 'value_n_t'), ('n_raw_tokens', 'value_n_r_t')]:

            if n_token_count in document_index.columns:
                try:
                    co_occurrences[target_field] = co_occurrences.value / float(
                        sum(document_index[n_token_count].values)
                    )
                except ZeroDivisionError:
                    co_occurrences[target_field] = 0.0
            else:
                logger.warning(f"{target_field}: cannot compute since {n_token_count} not in corpus document catalogue")

    co_occurrences['w1'] = co_occurrences.w1_id.apply(lambda x: id2token[x])
    co_occurrences['w2'] = co_occurrences.w2_id.apply(lambda x: id2token[x])

    if ignore_pad is not None:
        co_occurrences = co_occurrences[((co_occurrences.w1 != ignore_pad) & (co_occurrences.w2 != ignore_pad))]

    # FIXME #104 Co-occurrences.to_dataframe: Keep w1_id och w2_id + Token2Id. Drop w1 & w2
    # FIXME #105 Co-occurrences.to_dataframe: Drop value_n_d
    co_occurrences: CoOccurrenceDataFrame = co_occurrences[['w1', 'w2', 'value', 'value_n_d', 'value_n_t']]

    if transform_opts is not None:
        unique_tokens = set(co_occurrences.w1.unique().tolist()).union(co_occurrences.w2.unique().tolist())
        transform: TokensTransformer = TokensTransformer(transform_opts)
        keep_tokens = set(transform.transform(unique_tokens))
        co_occurrences = co_occurrences[(co_occurrences.w1.isin(keep_tokens)) & (co_occurrences.w2.isin(keep_tokens))]

    return co_occurrences
