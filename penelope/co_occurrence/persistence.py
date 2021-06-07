from __future__ import annotations

import contextlib
import json
import os
import pickle
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
from loguru import logger
from penelope.type_alias import CoOccurrenceDataFrame
from penelope.utility import read_json, replace_extension, right_chop, strip_path_and_extension

from ..corpus import (
    DocumentIndex,
    DocumentIndexHelper,
    ExtractTaggedTokensOpts,
    TextReaderOpts,
    Token2Id,
    TokensTransformOpts,
    VectorizedCorpus,
)
from .interface import ContextOpts, CoOccurrenceError

if TYPE_CHECKING:
    from .bundle import Bundle

jj = os.path.join

FILENAME_POSTFIX = '_co-occurrence.csv.zip'
FILENAME_PATTERN = f'*{FILENAME_POSTFIX}'
DOCUMENT_INDEX_POSTFIX = '_co-occurrence.document_index.zip'
DICTIONARY_POSTFIX = '_co-occurrence.dictionary.zip'
CORPUS_COUNTS_POSTFIX = '_corpus_windows_counts.pickle'
DOCUMENT_COUNTS_POSTFIX = '_document_windows_counts.npz'
VOCABULARY_MAPPING_POSTFIX = '_vocabs_mapping.pickle'


def to_folder_and_tag(filename: str, postfix: str = FILENAME_POSTFIX) -> Tuple[str, str]:
    """Strips out corpus folder and tag from filename having `postfix` ending"""
    folder, corpus_basename = os.path.split(filename)
    tag = right_chop(corpus_basename, postfix)
    return folder, tag


def to_filename(*, folder: str, tag: str, postfix: str = FILENAME_POSTFIX) -> str:
    return os.path.join(folder, f"{tag}{postfix}")


def _get_filename(tag: str, postfix: str) -> str:
    return f"{tag}{postfix}"


def _get_path(folder: str, tag: str, postfix: str) -> str:
    return to_filename(folder=folder, tag=tag, postfix=postfix)


def co_occurrence_filename(folder: str, tag: str) -> str:
    return _get_path(folder, tag, FILENAME_POSTFIX)


def document_index_filename(folder: str, tag: str) -> str:
    return _get_path(folder, tag, DOCUMENT_INDEX_POSTFIX)


def vocabulary_filename(folder: str, tag: str) -> str:
    return _get_path(folder, tag, DICTIONARY_POSTFIX)


def options_filename(folder: str, tag: str) -> str:
    return replace_extension(co_occurrence_filename(folder, tag), 'json')


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


# pylint: disable=redefined-outer-name
def store_co_occurrences(*, filename: str, co_occurrences: CoOccurrenceDataFrame, store_feather: bool = True) -> None:
    """Store co-occurrence result data to CSV-file (if loaded)"""
    if co_occurrences is None:
        return

    if filename.endswith('zip'):
        archive_name = f"{strip_path_and_extension(filename)}.csv"
        compression = dict(method='zip', archive_name=archive_name)
    else:
        compression = 'infer'

    logger.info("storing co-occurrences (CSV)")
    co_occurrences.to_csv(filename, sep='\t', header=True, compression=compression, decimal=',')

    if store_feather:
        with contextlib.suppress(Exception):
            logger.info("storing co-occurrences (feather)")
            co_occurrences.reset_index(drop=True).to_feather(replace_extension(filename, ".feather"), compression="lz4")


def load_co_occurrences(filename: str) -> CoOccurrenceDataFrame:
    """Load co-occurrences from CSV-file if exists on disk"""

    feather_filename: str = replace_extension(filename, ".feather")

    """ Read FEATHER if exists """
    if os.path.isfile(feather_filename):

        logger.info("loading FEATHER file")
        df: pd.DataFrame = pd.read_feather(feather_filename)
        if 'index' in df.columns:
            df.drop(columns='index', inplace=True)
        return df

    """ Read CSV if exists """
    if os.path.isfile(filename):

        co_occurrences: pd.DataFrame = pd.read_csv(filename, sep='\t', header=0, decimal=',', index_col=0)

        with contextlib.suppress(Exception):
            logger.info("caching to FEATHER file")
            store_feather(feather_filename, co_occurrences)

        return co_occurrences

    return None


def load_feather(filename: str) -> CoOccurrenceDataFrame:
    """Reads co-occurrences stored in Apache Arrow feather file format"""
    feather_filename: str = replace_extension(filename, ".feather")
    if os.path.isfile(feather_filename):
        co_occurrences: pd.DataFrame = pd.read_feather(feather_filename)
        return co_occurrences
    return None


def store_feather(filename: str, co_occurrences: CoOccurrenceDataFrame) -> None:
    """Stores co-occurrences in Apache Arrow feather file format"""
    feather_filename: str = replace_extension(filename, ".feather")
    co_occurrences = co_occurrences.reset_index()
    co_occurrences.to_feather(feather_filename, compression="lz4")


def load_document_index(folder: str, tag: str) -> DocumentIndex:
    path: str = to_filename(folder=folder, tag=tag, postfix=DOCUMENT_INDEX_POSTFIX)
    document_index = DocumentIndexHelper.load(path).document_index
    return document_index


def store_document_index(document_index: DocumentIndex, filename: str) -> None:
    DocumentIndexHelper(document_index).store(filename)


def load_vocabulary(folder: str, tag: str) -> Token2Id:
    path: str = to_filename(folder=folder, tag=tag, postfix=DICTIONARY_POSTFIX)
    token2id: Token2Id = Token2Id.load(path)
    return token2id


def load_options(filename: str) -> dict:
    """Loads co-occurrence compute options"""
    options_filename = replace_extension(filename, 'json')
    if os.path.isfile(options_filename):
        options = read_json(options_filename)
        return options
    return {'not_found': options_filename}


def store_vocabs_mapping(vocabs_mapping: Optional[Mapping[Tuple[int, int], int]], folder: str, tag: str):
    if vocabs_mapping:
        filename = to_filename(folder=folder, tag=tag, postfix=VOCABULARY_MAPPING_POSTFIX)
        with open(filename, 'wb') as fp:
            pickle.dump(vocabs_mapping, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_vocabs_mapping(folder: str, tag: str) -> Optional[Mapping[Tuple[int, int], int]]:
    filename = to_filename(folder=folder, tag=tag, postfix=VOCABULARY_MAPPING_POSTFIX)
    if os.path.isfile(filename):
        with open(filename, 'rb') as fp:
            vocabs_mapping: Mapping[Tuple[int, int], int] = pickle.load(fp)
            return vocabs_mapping
    return None


def store_options(*, options: dict, filename: str) -> None:
    """Also save options with same name as co-occurrence file"""
    with open(filename, 'w') as fp:
        json.dump(options, fp, indent=4, default=lambda _: '<not serializable>')


def create_options_bundle(
    *,
    reader_opts: TextReaderOpts,
    transform_opts: TokensTransformOpts,
    context_opts: ContextOpts,
    extract_opts: ExtractTaggedTokensOpts,
    **other_options,
):
    options = {
        **{
            'reader_opts': reader_opts.props,
            'context_opts': context_opts.props,
            'transform_opts': transform_opts.props,
            'extract_opts': extract_opts.props,
        },
        **other_options,
    }
    return options


@dataclass
class TokenWindowCountStatistics:

    """Corpus-wide tokens' window counts"""

    # FIXME Is this realy needed? Sum of matrix axis=0?
    corpus_counts: Mapping[int, int] = None

    """Document-level token window counts"""
    document_counts: scipy.sparse.spmatrix = None

    @staticmethod
    def load(folder: str, tag: str) -> "TokenWindowCountStatistics":
        return TokenWindowCountStatistics(
            corpus_counts=TokenWindowCountStatistics._load_corpus_counts(folder=folder, tag=tag),
            document_counts=TokenWindowCountStatistics._load_document_counts(folder=folder, tag=tag),
        )

    def store(self, folder: str, tag: str) -> None:
        self._store_corpus_counts(folder, tag)
        self._store_document_counts(folder, tag)

    def _store_corpus_counts(self, folder: str, tag: str):
        if self.corpus_counts:
            with open(self._corpus_counts_filename(folder, tag), 'wb') as fp:
                pickle.dump(self.corpus_counts, fp, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load_corpus_counts(folder: str, tag: str) -> Optional[Mapping[int, int]]:
        filename = to_filename(folder=folder, tag=tag, postfix=CORPUS_COUNTS_POSTFIX)
        if not os.path.isfile(filename):
            return None
        with open(filename, 'rb') as fp:
            counts: Counter = pickle.load(fp)
        return counts

    def _store_document_counts(self, folder: str, tag: str, compressed: bool = True) -> None:
        """Stores documents' (rows) token (column) window counts matrix"""
        filename = TokenWindowCountStatistics._document_counts_filename(folder, tag)
        if compressed:
            assert scipy.sparse.issparse(self.document_counts)
            scipy.sparse.save_npz(replace_extension(filename, '.npz'), self.document_counts, compressed=True)
        else:
            np.save(replace_extension(filename, '.npy'), self.document_counts, allow_pickle=True)

    @staticmethod
    def _load_document_counts(folder: str, tag: str) -> scipy.sparse.spmatrix:
        """Loads documents' (rows) token (column) window counts matrix"""
        filename = TokenWindowCountStatistics._document_counts_filename(folder, tag)
        if os.path.isfile(replace_extension(filename, '.npz')):
            return scipy.sparse.load_npz(replace_extension(filename, '.npz'))

        if os.path.isfile(replace_extension(filename, '.npy')):
            return np.load(replace_extension(filename, '.npy'), allow_pickle=True).item()

        return None

    @staticmethod
    def _corpus_counts_filename(folder: str, tag: str) -> str:
        return to_filename(folder=folder, tag=tag, postfix=CORPUS_COUNTS_POSTFIX)

    @staticmethod
    def _document_counts_filename(folder: str, tag: str) -> str:
        return to_filename(folder=folder, tag=tag, postfix=DOCUMENT_COUNTS_POSTFIX)


def store(bundle: "Bundle"):

    if not (bundle.tag and bundle.folder):
        raise CoOccurrenceError("store failed (folder and/or tag not specfied)")

    if bundle.corpus is None:
        raise CoOccurrenceError("store failed (corpus cannot be None)")

    if bundle.token2id is None:
        raise CoOccurrenceError("store failed (source token2id cannot be None)")

    folder, tag = bundle.folder, bundle.tag

    store_corpus(corpus=bundle.corpus, folder=folder, tag=tag, options=bundle.compute_options)
    store_document_index(bundle.document_index, document_index_filename(folder, tag))

    bundle.token2id.store(vocabulary_filename(folder, tag))
    bundle.window_counts.store(folder, tag)

    store_options(options=bundle.compute_options, filename=options_filename(folder, tag))
    store_co_occurrences(filename=co_occurrence_filename(folder, tag), co_occurrences=bundle.co_occurrences)

    store_vocabs_mapping(bundle.vocabs_mapping, folder, tag)


def load(filename: str = None, folder: str = None, tag: str = None, compute_frame: bool = True) -> dict:
    """Loads bundle identified by given filename i.e. `folder`/`tag`{FILENAME_POSTFIX}"""

    if not filename:
        if folder and tag:
            filename = to_filename(folder=folder, tag=tag)
        else:
            raise CoOccurrenceError("load: filename and folder/tag cannot both be empty")
    else:
        folder, tag = to_folder_and_tag(filename)

    corpus: VectorizedCorpus = load_corpus(folder, tag)
    token2id: Token2Id = load_vocabulary(folder, tag)
    document_index: DocumentIndex = load_document_index(folder, tag)
    options: dict = load_options(filename) or VectorizedCorpus.load_options(folder=folder, tag=tag)
    window_counts: TokenWindowCountStatistics = TokenWindowCountStatistics.load(folder, tag)
    vocabs_mapping: Optional[Mapping[Tuple[int, int], int]] = load_vocabs_mapping(folder=folder, tag=tag)

    corpus.remember_vocabs_mapping(vocabs_mapping)

    if token2id is None:
        raise CoOccurrenceError("Vocabulary is missing (corrupt data)!")

    if corpus is None:
        raise CoOccurrenceError("Co-occurrence corpus is missing (corrupt data)!")

    co_occurrences: CoOccurrenceDataFrame = load_co_occurrences(filename)

    if co_occurrences is None and compute_frame:
        co_occurrences = corpus.to_co_occurrences(token2id)

    bundle = dict(
        folder=folder,
        tag=tag,
        corpus=corpus,
        vocabs_mapping=vocabs_mapping,
        document_index=document_index,
        token2id=token2id,
        compute_options=options,
        window_counts=window_counts,
        co_occurrences=co_occurrences,
    )

    return bundle
