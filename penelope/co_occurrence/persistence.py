from __future__ import annotations

import contextlib
import json
import os
import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Mapping, Optional, Tuple, Type

import numpy as np
import pandas as pd
import scipy
from loguru import logger

from penelope.type_alias import CoOccurrenceDataFrame
from penelope.utility import create_instance, read_json, replace_extension, right_chop, strip_path_and_extension

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
DOCUMENT_COUNTS_POSTFIX = '_document_windows_counts.npz'
VOCABULARY_MAPPING_POSTFIX = '_vocabs_mapping.pickle'

CORPUS_COUNTS_POSTFIX = '_corpus_windows_counts.pickle'
CONCEPT_CORPUS_COUNTS_POSTFIX = '_concept_corpus_windows_counts.pickle'


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

    if options:
        corpus.dump_options(tag=tag, folder=folder, options=options)


def load_corpus(*, folder: str, tag: str) -> VectorizedCorpus:

    return (
        VectorizedCorpus.load(folder=folder, tag=tag) if VectorizedCorpus.dump_exists(folder=folder, tag=tag) else None
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

        df: pd.DataFrame = pd.read_feather(feather_filename)
        if 'index' in df.columns:
            df.drop(columns='index', inplace=True)

        return df

    """ Read CSV if exists """
    if os.path.isfile(filename):

        co_occurrences: pd.DataFrame = pd.read_csv(filename, sep='\t', header=0, decimal=',', index_col=0, engine='c')

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
    co_occurrences = co_occurrences.reset_index(drop=True)
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
    # def _store_cached(vocabs_mapping: dict, filename: str):

    #     with contextlib.suppress(Exception):
    #         df: pd.DataFrame = pd.DataFrame(data={'key': vocabs_mapping.keys(), 'value': vocabs_mapping.values()})
    #         df.to_feather(replace_extension(filename, "feather"))

    # def _load_cached(filename: str) -> None | dict:

    #     with contextlib.suppress(Exception):
    #         feather_filename: str = replace_extension(filename, "feather")
    #         if os.path.isfile(feather_filename):
    #             df: pd.DataFrame = pd.read_feather(feather_filename)
    #             return dict(zip(df.key.apply(tuple), df.value))

    #     return None

    filename: str = to_filename(folder=folder, tag=tag, postfix=VOCABULARY_MAPPING_POSTFIX)
    if os.path.isfile(filename):
        # vm: dict = _load_cached(filename)
        # if vm:
        #     return vm
        with open(filename, 'rb') as fp:
            vocabs_mapping: Mapping[Tuple[int, int], int] = pickle.load(fp)
            # _store_cached(vocabs_mapping, filename)
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
class WindowCountDTM:
    """Document Term Matrix that containing window counts for each token in each document"""

    dtm_wc: scipy.sparse.spmatrix = None

    # @property
    # def total_term_window_counts(self):
    #     """Corpus-wide tokens' window counts"""
    #     return self.dtm_wc.sum(axis=0).A1

    def slice(self, keep_token_ids: List(int), inplace=True) -> WindowCountDTM:

        if len(keep_token_ids) == self.dtm_wc.shape[1]:
            return self

        matrix: scipy.sparse.spmatrix = self.dtm_wc[:, keep_token_ids]

        if inplace:
            self.dtm_wc = matrix
            return self

        return WindowCountDTM(dtm_wc=matrix)

    def store(self, folder: str, tag: str, compressed: bool = True) -> None:
        """Stores documents' (rows) token (column) window counts matrix"""
        filename = to_filename(folder=folder, tag=tag, postfix=DOCUMENT_COUNTS_POSTFIX)
        if compressed:
            assert scipy.sparse.issparse(self.dtm_wc)
            scipy.sparse.save_npz(replace_extension(filename, '.npz'), self.dtm_wc, compressed=True)
        else:
            np.save(replace_extension(filename, '.npy'), self.dtm_wc, allow_pickle=True)

    @staticmethod
    def load(folder: str, tag: str) -> "WindowCountDTM":
        """Loads documents' (rows) token (column) window counts matrix"""
        matrix: scipy.sparse.spmatrix = None
        filename = to_filename(folder=folder, tag=tag, postfix=DOCUMENT_COUNTS_POSTFIX)
        if os.path.isfile(replace_extension(filename, '.npz')):
            matrix = scipy.sparse.load_npz(replace_extension(filename, '.npz'))

        if os.path.isfile(replace_extension(filename, '.npy')):
            matrix = np.load(replace_extension(filename, '.npy'), allow_pickle=True).item()

        return WindowCountDTM(dtm_wc=matrix)

    @property
    def corpus_term_window_counts0(self):
        return self.dtm_wc.sum(axis=0).A1


def store(bundle: "Bundle"):

    if not (bundle.tag and bundle.folder):
        raise CoOccurrenceError("store failed (folder and/or tag not specfied)")

    if bundle.corpus is None:
        raise CoOccurrenceError("store failed (corpus cannot be None)")

    if bundle.token2id is None:
        raise CoOccurrenceError("store failed (source token2id cannot be None)")

    folder, tag = bundle.folder, bundle.tag

    os.makedirs(folder, exist_ok=True)

    store_corpus(corpus=bundle.corpus, folder=folder, tag=tag, options=bundle.compute_options)
    bundle.corpus.window_counts.store(folder, tag)

    if bundle.concept_corpus:
        store_corpus(corpus=bundle.concept_corpus, folder=folder, tag=tag + "_concept", options=bundle.compute_options)
        bundle.concept_corpus.window_counts.store(folder, tag + "_concept")

    store_document_index(bundle.document_index, document_index_filename(folder, tag))

    bundle.token2id.store(vocabulary_filename(folder, tag))

    store_options(options=bundle.compute_options, filename=options_filename(folder, tag))
    store_co_occurrences(filename=co_occurrence_filename(folder, tag), co_occurrences=bundle.co_occurrences)

    store_vocabs_mapping(bundle.token_ids_2_pair_id, folder, tag)


def load(filename: str = None, folder: str = None, tag: str = None, compute_frame: bool = True) -> Bundle:
    """Loads bundle identified by given filename i.e. `folder`/`tag`{FILENAME_POSTFIX}"""

    if not filename:
        if folder and tag:
            filename = to_filename(folder=folder, tag=tag)
        else:
            raise CoOccurrenceError("load: filename and folder/tag cannot both be empty")
    else:
        folder, tag = to_folder_and_tag(filename)

    options: dict = load_options(filename) or VectorizedCorpus.load_options(folder=folder, tag=tag)
    vocabs_mapping: Optional[Mapping[Tuple[int, int], int]] = load_vocabs_mapping(folder=folder, tag=tag)

    corpus: VectorizedCorpus = load_corpus(folder=folder, tag=tag).remember(
        window_counts=WindowCountDTM.load(folder, tag), vocabs_mapping=vocabs_mapping
    )

    concept_corpus: VectorizedCorpus = load_corpus(folder=folder, tag=tag + "_concept")
    if concept_corpus:
        concept_corpus.remember(
            window_counts=WindowCountDTM.load(folder=folder, tag=tag + "_concept"),
            vocabs_mapping=vocabs_mapping,
        )

    token2id: Token2Id = load_vocabulary(folder, tag)
    document_index: DocumentIndex = load_document_index(folder, tag)

    if token2id is None:
        raise CoOccurrenceError("Vocabulary is missing (corrupt data)!")

    if corpus is None:
        raise CoOccurrenceError("Co-occurrence corpus is missing (corrupt data)!")

    co_occurrences: CoOccurrenceDataFrame = load_co_occurrences(filename)

    if co_occurrences is None and compute_frame:
        co_occurrences = corpus.to_co_occurrences(token2id)

    bundle_cls: Type["Bundle"] = create_instance('penelope.co_occurrence.bundle.Bundle')
    bundle: "Bundle" = bundle_cls(
        folder=folder,
        tag=tag,
        corpus=corpus,
        concept_corpus=concept_corpus,
        vocabs_mapping=vocabs_mapping,
        document_index=document_index,
        token2id=token2id,
        compute_options=options,
        co_occurrences=co_occurrences,
    )
    return bundle
