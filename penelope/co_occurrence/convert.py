import contextlib
import json
import os
from dataclasses import dataclass
from typing import Mapping, Tuple, Union

import pandas as pd
import scipy
from loguru import logger
from penelope.corpus import (
    CorpusVectorizer,
    DocumentIndex,
    DocumentIndexHelper,
    ITokenizedCorpus,
    Token2Id,
    TokenizedCorpus,
    TokensTransformer,
    TokensTransformOpts,
    VectorizedCorpus,
)
from penelope.corpus.readers import ExtractTaggedTokensOpts, ICorpusReader, TextReaderOpts
from penelope.notebook.word_trends import TrendsData
from penelope.utility import read_json, replace_extension, right_chop, strip_path_and_extension

from .interface import ContextOpts, CoOccurrenceError

CO_OCCURRENCE_FILENAME_POSTFIX = '_co-occurrence.csv.zip'
CO_OCCURRENCE_FILENAME_PATTERN = f'*{CO_OCCURRENCE_FILENAME_POSTFIX}'

CoOccurrenceDataFrame = pd.DataFrame


def filename_to_folder_and_tag(co_occurrences_filename: str) -> Tuple[str, str]:
    """Strips out corpus folder and tag from filename having CO_OCCURRENCE_FILENAME_POSTFIX ending"""
    corpus_folder, corpus_basename = os.path.split(co_occurrences_filename)
    corpus_tag = right_chop(corpus_basename, CO_OCCURRENCE_FILENAME_POSTFIX)
    return corpus_folder, corpus_tag


def tag_to_filename(*, tag: str) -> str:
    return f"{tag}{CO_OCCURRENCE_FILENAME_POSTFIX}"


def folder_and_tag_to_filename(*, folder: str, tag: str) -> str:
    return os.path.join(folder, tag_to_filename(tag=tag))


def store_co_occurrences(filename: str, co_occurrences: CoOccurrenceDataFrame):
    """Store co-occurrence result data to CSV-file"""

    if filename.endswith('zip'):
        archive_name = f"{strip_path_and_extension(filename)}.csv"
        compression = dict(method='zip', archive_name=archive_name)
    else:
        compression = 'infer'

    logger.info("storing CSV file")
    co_occurrences.to_csv(filename, sep='\t', header=True, compression=compression, decimal=',')

    # with contextlib.suppress(Exception):
    try:
        logger.info("storing FEATHER file")
        co_occurrences.reset_index(drop=True).to_feather(replace_extension(filename, ".feather"), compression="lz4")
    except Exception as ex:
        logger.info("store as FEATHER file failed")
        logger.error(ex)


def load_co_occurrences(filename: str) -> CoOccurrenceDataFrame:
    """Load co-occurrences from CSV-file"""

    feather_filename: str = replace_extension(filename, ".feather")

    """ Read FEATHER if exists """
    if os.path.isfile(feather_filename):
        logger.info("loading FEATHER file")
        df: pd.DataFrame = pd.read_feather(feather_filename)
        if 'index' in df.columns:
            df.drop(columns='index', inplace=True)
        return df

    co_occurrences: pd.DataFrame = pd.read_csv(filename, sep='\t', header=0, decimal=',', index_col=0)

    with contextlib.suppress(Exception):
        logger.info("caching to FEATHER file")
        store_feather(feather_filename, co_occurrences)

    return co_occurrences


def load_feather(filename: str) -> CoOccurrenceDataFrame:
    feather_filename: str = replace_extension(filename, ".feather")
    if os.path.isfile(feather_filename):
        logger.info("loading FEATHER file")
        co_occurrences: pd.DataFrame = pd.read_feather(feather_filename)
        logger.info(f"COLUMNS (after load): {', '.join(co_occurrences.columns.tolist())}")
        return co_occurrences
    logger.info("FEATHER load FAILED")
    return None


def store_feather(filename: str, co_occurrences: CoOccurrenceDataFrame) -> None:
    feather_filename: str = replace_extension(filename, ".feather")
    # with contextlib.suppress(Exception):
    #     logger.info("caching to FEATHER file")
    logger.info(f"COLUMNS: {', '.join(co_occurrences.columns.tolist())}")
    co_occurrences = co_occurrences.reset_index()
    co_occurrences.to_feather(feather_filename, compression="lz4")
    logger.info(f"COLUMNS (after reset): {', '.join(co_occurrences.columns.tolist())}")


class PartitionKeyNotUniqueKey(ValueError):
    ...


def to_vectorized_corpus(
    *,
    co_occurrences: CoOccurrenceDataFrame,
    document_index: DocumentIndex,
    value_key: str,
    partition_key: Union[int, str],
) -> VectorizedCorpus:
    """Creates a DTM corpus from a co-occurrence result set that was partitioned by `partition_column`."""

    if document_index[partition_key].unique().size != len(document_index):
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
    )

    document_index = document_index.set_index('document_id', drop=False).rename_axis('').sort_index()

    v_corpus = VectorizedCorpus(coo_matrix, token2id=token2id, document_index=document_index)

    return v_corpus


def to_co_occurrence_matrix(
    corpus_or_reader: Union[ICorpusReader, TokenizedCorpus], vocabulary: Mapping[str, int] = None
) -> scipy.sparse.spmatrix:
    """Computes a term-term co-ocurrence matrix for documents in corpus/reader.

    Parameters
    ----------
    corpus_or_reader : Union[ICorpusReader,TokenizedCorpus]
        Sequence of tokenized documents

    Returns
    -------
    pd.DataFrame
        Upper diagonal of term-term frequency matrix (TTM). Note that diagonal (wi, wi) is not returned
    """

    if not isinstance(corpus_or_reader, ITokenizedCorpus):
        corpus_or_reader = TokenizedCorpus(reader=corpus_or_reader)

    vocabulary = vocabulary or corpus_or_reader.token2id
    dtm_corpus: VectorizedCorpus = CorpusVectorizer().fit_transform(
        corpus_or_reader, already_tokenized=True, vocabulary=vocabulary
    )
    term_term_matrix = dtm_corpus.co_occurrence_matrix()
    return term_term_matrix


def to_dataframe(
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


@dataclass
class Bundle:

    co_occurrences_filename: str = None
    corpus_folder: str = None
    corpus_tag: str = None

    co_occurrences: pd.DataFrame = None
    document_index: DocumentIndex = None
    token2id: Token2Id = None

    corpus: VectorizedCorpus = None
    compute_options: dict = None


def store_bundle(output_filename: str, bundle: Bundle) -> Bundle:

    store_co_occurrences(output_filename, bundle.co_occurrences)

    if bundle.corpus is not None:

        if bundle.corpus_tag is None:
            bundle.corpus_tag = strip_path_and_extension(output_filename)

        bundle.corpus_folder = os.path.split(output_filename)[0]
        bundle.corpus.dump(tag=bundle.corpus_tag, folder=bundle.corpus_folder)
        bundle.corpus.dump_options(
            tag=bundle.corpus_tag,
            folder=bundle.corpus_folder,
            options=bundle.compute_options,
        )

        if bundle.token2id is not None:
            bundle.token2id.store(os.path.join(bundle.corpus_folder, "dictionary.zip"))

        DocumentIndexHelper(bundle.document_index).store(
            os.path.join(bundle.corpus_folder, f"{bundle.corpus_tag}_document_index.csv")
        )

    """Also save options with same name as co-occurrence file"""
    with open(replace_extension(output_filename, 'json'), 'w') as json_file:
        json.dump(bundle.compute_options, json_file, indent=4)


def load_bundle(co_occurrences_filename: str, compute_corpus: bool = True) -> "Bundle":
    """Loads bundle identified by given filename i.e. `corpus_folder`/`corpus_tag`_co-occurrence.csv.zip"""

    corpus_folder, corpus_tag = filename_to_folder_and_tag(co_occurrences_filename)
    co_occurrences = load_co_occurrences(co_occurrences_filename)
    document_index = DocumentIndexHelper.load(
        os.path.join(corpus_folder, f"{corpus_tag}_document_index.csv")
    ).document_index
    corpus = (
        VectorizedCorpus.load(folder=corpus_folder, tag=corpus_tag)
        if VectorizedCorpus.dump_exists(folder=corpus_folder, tag=corpus_tag)
        else None
    )
    corpus_options: dict = VectorizedCorpus.load_options(folder=corpus_folder, tag=corpus_tag)
    options = load_options(co_occurrences_filename) or corpus_options

    token2id: Token2Id = Token2Id.load(os.path.join(corpus_folder, "dictionary.zip"))

    if token2id is None:
        logger.info("no vocabulary in bundle (creating a new vocabularyfrom co-occurrences)")
        token2id = Token2Id()
        token2id.ingest(co_occurrences.w1)
        token2id.ingest(co_occurrences.w2)

    if corpus is None and compute_corpus:

        if len(options.get('partition_keys', []) or []) == 0:
            raise ValueError("load_bundle: cannot load, unknown partition key")

        partition_key = options['partition_keys'][0]
        corpus = to_vectorized_corpus(
            co_occurrences=co_occurrences,
            document_index=document_index,
            value_key='value',
            partition_key=partition_key,
        )

    bundle = Bundle(
        co_occurrences_filename=co_occurrences_filename,
        corpus_folder=corpus_folder,
        corpus_tag=corpus_tag,
        co_occurrences=co_occurrences,
        document_index=document_index,
        token2id=token2id,
        compute_options=options,
        corpus=corpus,
    )

    return bundle


def to_trends_data(bundle: Bundle, n_count=25000):

    trends_data = TrendsData(
        compute_options=bundle.compute_options,
        corpus=bundle.corpus,
        corpus_folder=bundle.corpus_folder,
        corpus_tag=bundle.corpus_tag,
        n_count=n_count,
    ).remember(co_occurrences=bundle.co_occurrences, document_index=bundle.document_index)

    return trends_data


def load_options(co_occurrences_filename: str) -> dict:
    """Loads co-occurrence compute options"""
    options_filename = replace_extension(co_occurrences_filename, 'json')
    if os.path.isfile(options_filename):
        options = read_json(options_filename)
        return options
    return {'not_found': options_filename}


def create_options_bundle(
    *,
    reader_opts: TextReaderOpts,
    tokens_transform_opts: TokensTransformOpts,
    context_opts: ContextOpts,
    extract_tokens_opts: ExtractTaggedTokensOpts,
    **other_options,
):
    options = {
        **{
            'reader_opts': reader_opts.props,
            'context_opts': context_opts.props,
            'tokens_transform_opts': tokens_transform_opts.props,
            'extract_tokens_opts': extract_tokens_opts.props,
        },
        **other_options,
    }
    return options
