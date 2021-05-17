import contextlib
import json
import os
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from loguru import logger
from penelope.corpus import (
    DocumentIndex,
    DocumentIndexHelper,
    ExtractTaggedTokensOpts,
    TextReaderOpts,
    Token2Id,
    TokensTransformOpts,
    VectorizedCorpus,
)
from penelope.type_alias import CoOccurrenceDataFrame
from penelope.utility import read_json, replace_extension, right_chop, strip_path_and_extension

from .interface import ContextOpts

CO_OCCURRENCE_FILENAME_POSTFIX = '_co-occurrence.csv.zip'
CO_OCCURRENCE_FILENAME_PATTERN = f'*{CO_OCCURRENCE_FILENAME_POSTFIX}'


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

    # FIXME: This patch should be deprecated (Token2Id must exists!)
    if token2id is None and 'w1' in co_occurrences.columns:
        logger.info("no vocabulary in bundle (creating a new vocabulary from co-occurrences)")
        token2id = Token2Id()
        token2id.ingest(co_occurrences.w1)
        token2id.ingest(co_occurrences.w2)

    if corpus is None and compute_corpus:
        raise ValueError("Compute of corpus during load is disabled")
        # if len(options.get('partition_keys', []) or []) == 0:
        #     raise ValueError("load_bundle: cannot load, unknown partition key")

        # partition_key = options['partition_keys'][0]
        # corpus = to_vectorized_corpus(
        #     co_occurrences=co_occurrences,
        #     document_index=document_index,
        #     partition_key=partition_key,
        # )

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
