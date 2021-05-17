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

from .interface import ContextOpts, CoOccurrenceError

jj = os.path.join

FILENAME_POSTFIX = '_co-occurrence.csv.zip'
FILENAME_PATTERN = f'*{FILENAME_POSTFIX}'
DOCUMENT_INDEX_POSTFIX = '_co-occurrence.document_index.zip'
DICTIONARY_POSTFIX = '_co-occurrence.dictionary.zip'


def to_folder_and_tag(filename: str, postfix: str = FILENAME_POSTFIX) -> Tuple[str, str]:
    """Strips out corpus folder and tag from filename having `postfix` ending"""
    folder, corpus_basename = os.path.split(filename)
    tag = right_chop(corpus_basename, postfix)
    return folder, tag


def to_filename(*, folder: str, tag: str, postfix: str = FILENAME_POSTFIX) -> str:
    return os.path.join(folder, f"{tag}{postfix}")


@dataclass
class Bundle:

    folder: str = None
    tag: str = None

    co_occurrences: pd.DataFrame = None
    document_index: DocumentIndex = None
    token2id: Token2Id = None

    corpus: VectorizedCorpus = None
    compute_options: dict = None

    def _get_filename(self, postfix: str) -> str:
        return f"{self.tag}{postfix}"

    def _get_path(self, postfix: str) -> str:
        return to_filename(folder=self.folder, tag=self.tag, postfix=postfix)

    @property
    def co_occurrence_filename(self) -> str:
        return self._get_path(FILENAME_POSTFIX)

    @property
    def document_index_filename(self) -> str:
        return self._get_path(DOCUMENT_INDEX_POSTFIX)

    @property
    def dictionary_filename(self) -> str:
        return self._get_path(DICTIONARY_POSTFIX)

    @property
    def options_filename(self) -> str:
        return replace_extension(self.co_occurrence_filename, 'json')

    def store(self, *, folder: str = None, tag: str = None) -> "Bundle":

        if tag and folder:
            self.tag, self.folder = tag, folder

        if not (self.tag and self.folder):
            raise CoOccurrenceError("store failed (folder and/or tag not specfied)")

        store_co_occurrences(self.co_occurrence_filename, self.co_occurrences)
        store_corpus(corpus=self.corpus, folder=self.folder, tag=self.tag, options=self.compute_options)

        self.token2id.store(self.dictionary_filename)

        DocumentIndexHelper(self.document_index).store(self.document_index_filename)

        """Also save options with same name as co-occurrence file"""
        with open(self.options_filename, 'w') as json_file:
            json.dump(self.compute_options, json_file, indent=4)

        return self

    @staticmethod
    def load(filename: str = None, folder: str = None, tag: str = None, compute_corpus: bool = True) -> "Bundle":
        """Loads bundle identified by given filename i.e. `folder`/`tag`{FILENAME_POSTFIX}"""

        if filename:
            folder, tag = to_folder_and_tag(filename)
        elif folder and tag:
            filename = to_filename(folder=folder, tag=tag)
        else:
            raise CoOccurrenceError("load: filename and folder/tag cannot both be empty")

        co_occurrences: CoOccurrenceDataFrame = load_co_occurrences(filename)
        document_index: DocumentIndex = load_document_index(folder, tag)
        corpus: VectorizedCorpus = load_corpus(folder, tag)
        corpus_options: dict = VectorizedCorpus.load_options(folder=folder, tag=tag)
        options: dict = load_options(filename) or corpus_options
        token2id: Token2Id = load_dictionary(folder, tag)

        if token2id is None:
            raise CoOccurrenceError("Dictionary is missing - please reprocess setup!")

        if corpus is None and compute_corpus:
            raise ValueError("Compute of corpus during load is disabled")
            # corpus = to_vectorized_corpus(
            #     co_occurrences=co_occurrences,
            #     document_index=document_index,
            #     token2id=token2id,
            # )

        bundle = Bundle(
            folder=folder,
            tag=tag,
            co_occurrences=co_occurrences,
            document_index=document_index,
            token2id=token2id,
            compute_options=options,
            corpus=corpus,
        )

        return bundle


def store_corpus(*, corpus: VectorizedCorpus, folder: str, tag: str, options: dict) -> None:

    if corpus is None:
        return

    corpus.dump(tag=tag, folder=folder)
    corpus.dump_options(tag=tag, folder=folder, options=options)


def load_corpus(corpus_folder: str, corpus_tag: str) -> VectorizedCorpus:
    return (
        VectorizedCorpus.load(folder=corpus_folder, tag=corpus_tag)
        if VectorizedCorpus.dump_exists(folder=corpus_folder, tag=corpus_tag)
        else None
    )


def store_co_occurrences(filename: str, co_occurrences: CoOccurrenceDataFrame) -> None:
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
        co_occurrences: pd.DataFrame = pd.read_feather(feather_filename)
        return co_occurrences
    return None


def store_feather(filename: str, co_occurrences: CoOccurrenceDataFrame) -> None:
    feather_filename: str = replace_extension(filename, ".feather")
    co_occurrences = co_occurrences.reset_index()
    co_occurrences.to_feather(feather_filename, compression="lz4")


def load_document_index(folder: str, tag: str) -> DocumentIndex:
    path: str = to_filename(folder=folder, tag=tag, postfix=DOCUMENT_INDEX_POSTFIX)
    document_index = DocumentIndexHelper.load(path).document_index
    return document_index


def store_document_index(document_index: DocumentIndex, filename: str) -> None:
    DocumentIndexHelper(document_index).store(filename)


def load_dictionary(folder: str, tag: str) -> Token2Id:
    path: str = to_filename(folder=folder, tag=tag, postfix=DICTIONARY_POSTFIX)
    token2id: Token2Id = Token2Id.load(path)
    return token2id


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
