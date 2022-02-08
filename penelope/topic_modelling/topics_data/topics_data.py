from __future__ import annotations

import os
import pickle
import types
from functools import cached_property
from os.path import isfile
from os.path import join as jj
from typing import Callable, List, Protocol, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from penelope import corpus as pc
from penelope import utility as pu

from . import token as tt
from .document import DocumentTopicsCalculator

CSV_OPTS: dict = dict(sep='\t', header=0, index_col=0, na_filter=False)

# pylint: disable=too-many-public-methods,access-member-before-definition,attribute-defined-outside-init,no-member


def smart_load(filename: str, feather_pipe: Callable[[pd.DataFrame], pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
    feather_filename: str = pu.replace_extension(filename, "feather")
    if isfile(feather_filename):
        data: pd.DataFrame = pd.read_feather(feather_filename)
        if feather_pipe is not None:
            data = data.pipe(feather_pipe, **kwargs)
    elif isfile(filename):
        data = pd.read_csv(filename, **CSV_OPTS)
    else:
        raise FileNotFoundError(f"{filename}")
    return data


class MemoryUsageMixIn:
    def memory_usage(self, total: bool = True) -> dict:
        return {
            "document_index": pu.size_of(self.document_index, unit='MB', total=total),
            "dictionary": pu.size_of(self.dictionary, unit='MB', total=total),
            "topic_token_weights": pu.size_of(self.topic_token_weights, unit='MB', total=total),
            "topic_token_overview": pu.size_of(self.topic_token_overview, unit='MB', total=total),
            "document_topic_weights": pu.size_of(self.document_topic_weights, unit='MB', total=total),
        }

    def log_usage(self, total: bool = False, verbose: bool = True) -> None:
        usage: dict = self.memory_usage(total=total)
        if not verbose and total:
            sw: str = ', '.join([f"{k}: {v}" for k, v in usage.items()])
            logger.info(f"{sw}")
        else:
            for k, v in usage.items():
                if isinstance(v, dict):
                    sw: str = ', '.join([f"{c}: {w}" for c, w in v.items()])
                    logger.info(f"{k}: {sw}")
                else:
                    logger.info(f"{k}: {v}")


class SlimItMixIn:
    def slim_types(self) -> InferredTopicsData:

        """document_index"""
        self.document_index['year'] = self.document_index['year'].astype(np.int16)
        self.document_index['n_tokens'] = self.document_index['n_tokens'].astype(np.int32)
        self.document_index['n_raw_tokens'] = self.document_index['n_raw_tokens'].astype(np.int32)
        if 'document_id' in self.document_index.columns:
            self.document_index['document_id'] = self.document_index['document_id'].astype(np.int32)
        for column in set(pu.PD_PoS_tag_groups.index.to_list()).intersection(self.document_index.columns):
            self.document_index[column] = self.document_index[column].astype(np.int32)

        """dictionary"""

        """topic_token_weights"""
        self.topic_token_weights['topic_id'] = self.topic_token_weights['topic_id'].astype(np.int16)
        self.topic_token_weights['token_id'] = self.topic_token_weights['token_id'].astype(np.int32)
        self.topic_token_weights['weight'] = self.topic_token_weights['weight'].astype(np.float32)

        """document_topic_weights"""
        self.document_topic_weights['document_id'] = self.document_topic_weights['document_id'].astype(np.int32)
        self.document_topic_weights['topic_id'] = self.document_topic_weights['topic_id'].astype(np.int16)
        self.document_topic_weights['weight'] = self.document_topic_weights['weight'].astype(np.float32)
        self.document_topic_weights['year'] = self.document_topic_weights['year'].astype(np.int16)

        self.topic_token_overview['label'] = self.topic_token_overview['label'].astype(str)

        return self

    def slimmer(self) -> InferredTopicsData:

        """document_index"""
        remove_columns = set(pu.PD_PoS_tag_groups.index.to_list()) | {'filename', 'year2', 'number'}
        self.document_index.drop(columns=list(remove_columns.intersection(self.document_index.columns)), inplace=True)
        if 'document_id' in self.document_index.columns:
            self.document_index.set_index('document_id', drop=True, inplace=True)

        """dictionary"""
        if 'dfs' in self.dictionary.columns:
            self.dictionary.drop(columns='dfs', inplace=True, errors='ignore')

        """topic_token_weights"""
        if 'token_id' not in self.topic_token_weights.columns:
            self.topic_token_weights = self.topic_token_weights.reset_index().set_index('topic_id')

        if 'token' in self.topic_token_weights:
            self.topic_token_weights.drop(columns='token', inplace=True)

        # FIXME #149 Varför är weights `topic_token_weights` så stora tal???

        """document_topic_weights"""

        return self


class InferredTopicsData(SlimItMixIn, MemoryUsageMixIn, tt.TopicTokensMixIn):
    """The result of applying a topic model on a corpus.
    The content, a generic set of pd.DataFrames, is common to all types of model engines.
    """

    def __init__(
        self,
        *,
        dictionary: pd.DataFrame,
        topic_token_weights: pd.DataFrame,
        topic_token_overview: pd.DataFrame,
        document_index: pd.DataFrame,
        document_topic_weights: pd.DataFrame,
    ):
        """Container for predicted topics data."""
        super().__init__()

        self.dictionary: pd.DataFrame = dictionary
        self.document_index: pd.DataFrame = document_index
        self.topic_token_weights: pd.DataFrame = topic_token_weights
        self.document_topic_weights: pd.DataFrame = pc.DocumentIndexHelper(document_index).overload(
            document_topic_weights, 'year'
        )
        self.topic_token_overview: pd.DataFrame = topic_token_overview
        self.calculator = DocumentTopicsCalculator(self)

        if 'token' not in self.topic_token_weights.columns:
            self.topic_token_weights['token'] = self.topic_token_weights['token_id'].apply(self.id2term.get)

    @property
    def num_topics(self) -> int:
        return int(self.topic_token_overview.index.max()) + 1

    @property
    def n_topics(self) -> int:
        return self.num_topics

    @property
    def timespan(self) -> Tuple[int, int]:
        return self.year_period

    def startspan(self, n: int) -> Tuple[int, int]:
        return (self.year_period[0], min(self.year_period[1], self.year_period[0] + n))

    def stopspan(self, n: int) -> Tuple[int, int]:
        return (max(self.year_period[0], self.year_period[1] - n), self.year_period[1])

    @property
    def year_period(self) -> Tuple[int, int]:
        """Returns documents `year` interval (if exists)"""
        if 'year' not in self.document_topic_weights.columns:
            return (None, None)
        return (self.document_topic_weights.year.min(), self.document_topic_weights.year.max())

    @property
    def topic_ids(self) -> List[int]:
        """Returns unique topic ids """
        return list(self.document_topic_weights.topic_id.unique())

    @cached_property
    def id2term(self) -> dict:
        # return self.dictionary.token.to_dict()
        return {i: t for i, t in zip(self.dictionary.index, self.dictionary.token)}

    @cached_property
    def term2id(self) -> dict:
        # return {v: k for k, v in self.id2term.items()}
        return {t: i for i, t in zip(self.dictionary.index, self.dictionary.token)}

    @property
    def id2token(self) -> dict:
        return self.id2term

    @cached_property
    def token2id(self) -> pc.Token2Id:
        return pc.Token2Id(data=self.term2id)

    def store(self, target_folder: str, pickled: bool = False, feather: bool = True):
        """Stores topics data in `target_folder` either as pickled file or individual zipped files """
        os.makedirs(target_folder, exist_ok=True)
        if pickled:
            PickleUtility.store(self, target_folder=target_folder)
        else:
            self._store_csv(target_folder)
            if feather:
                self._store_feather(target_folder)

    def _store_csv(self, target_folder: str) -> None:

        data: list[tuple[pd.DataFrame, str]] = [
            (self.document_index.rename_axis(''), 'documents.csv'),
            (self.dictionary, 'dictionary.csv'),
            (self.topic_token_weights, 'topic_token_weights.csv'),
            (self.topic_token_overview, 'topic_token_overview.csv'),
            (self.document_topic_weights, 'document_topic_weights.csv'),
        ]

        for (df, name) in data:
            archive_name = jj(target_folder, pu.replace_extension(name, ".zip"))
            pu.pandas_to_csv_zip(archive_name, (df, name), extension="csv", sep='\t')

    def _store_feather(self, target_folder: str) -> None:

        self.dictionary.reset_index().to_feather(jj(target_folder, "dictionary.feather"))
        self.document_index.reset_index(drop=True).to_feather(jj(target_folder, "documents.feather"))
        self.topic_token_weights.reset_index(drop=True).to_feather(jj(target_folder, "topic_token_weights.feather"))
        self.document_topic_weights.reset_index(drop=True).to_feather(
            jj(target_folder, "document_topic_weights.feather")
        )
        self.topic_token_overview.reset_index().to_feather(jj(target_folder, "topic_token_overview.feather"))

    @staticmethod
    def load(
        *,
        folder: str,
        filename_fields: pu.FilenameFieldSpecs = None,
        slim: bool = False,
        verbose: bool = False,
    ):
        """Loads previously stored aggregate"""

        if not isfile(jj(folder, "topic_token_weights.zip")):
            return PickleUtility.explode(source=folder, target_folder=folder)

        document_index: pd.DataFrame = (
            pd.read_feather(jj(folder, "documents.feather")).rename_axis('document_id')
            if isfile(jj(folder, "documents.feather"))
            else pc.load_document_index(
                jj(folder, 'documents.zip'), filename_fields=filename_fields, **CSV_OPTS
            ).set_index('document_id', drop=True)
        )

        data: InferredTopicsData = InferredTopicsData(
            dictionary=smart_load(jj(folder, 'dictionary.zip'), pu.set_index, columns='token_id'),
            document_index=document_index,
            topic_token_weights=smart_load(jj(folder, 'topic_token_weights.zip')),
            document_topic_weights=smart_load(jj(folder, 'document_topic_weights.zip')),
            topic_token_overview=smart_load(jj(folder, 'topic_token_overview.zip'), pu.set_index, columns='topic_id'),
        )

        # HACK: Handle renamed column:
        data.document_index = fix_renamed_columns(data.document_index)
        assert "year" in data.document_index.columns

        data.topic_token_overview = data.load_topic_labels(folder, **CSV_OPTS)

        data.slim_types()
        if slim:
            data.slimmer()
        if verbose:
            data.log_usage(total=True)
        return data

    def load_topic_labels(self, folder: str, **csv_opts: dict) -> pd.DataFrame:

        tto: pd.DataFrame = self.topic_token_overview
        if isfile(jj(folder, "topic_token_overview_label.csv")):
            labeled_tto: pd.DataFrame = pd.read_csv(jj(folder, 'topic_token_overview_label.csv'), **csv_opts)
            if self.is_satisfied_topic_token_overview(labeled_tto):
                # logger.info(f"labeled file loaded from: {folder}")
                tto = labeled_tto

        if 'label' not in tto.columns:
            tto['label'] = tto['document_id'].astype(str) if 'document_id' in tto.columns else tto.index.astype(str)

        return tto

    def is_satisfied_topic_token_overview(self, labeled_overview: pd.DataFrame) -> bool:
        try:
            overview: pd.DataFrame = self.topic_token_overview
            if len(labeled_overview) != len(overview):
                raise ValueError(f"length not {len(overview)} as expected")
            if 'label' not in labeled_overview.columns:
                raise ValueError("label column is missing")
            if (labeled_overview.index != overview.index).any():
                raise ValueError("index (topic_id) mismatch")
        except ValueError as ex:
            logger.warning(f"skipping labeled file: {ex}")
            return False
        return True

    @staticmethod
    def load_token2id(folder: str) -> pc.Token2Id:
        dictionary: pd.DataFrame = smart_load(jj(folder, 'dictionary.zip'), pu.set_index, columns='token_id')
        token2id: pc.Token2Id = pc.Token2Id(data={t: i for (t, i) in zip(dictionary.token, dictionary.index)})
        return token2id

    @property
    def topic_labels(self) -> dict:
        if 'label' not in self.topic_token_overview.columns:
            return {}
        return self.topic_token_overview['label'].to_dict()


def fix_renamed_columns(di: pd.DataFrame) -> pd.DataFrame:
    """Add count columns `n_tokens` and `n_rws_tokens" if missing and other/renamed column exists."""
    for missing in {"n_tokens", "n_raw_tokens"} - set(di.columns):
        for existing in {"n_terms", "n_tokens", "n_raw_tokens"}.intersection(di.columns):
            di[missing] = di[existing]
            break
    if 'n_terms' in di.columns:
        di = di.drop(columns='n_terms')
    return di


# FXIME: Reprecate pickled stora
class PickleUtility:
    ...

    @staticmethod
    def load(folder: str) -> InferredTopicsData:

        if not isfile(jj(folder, "inferred_topics.pickle")):
            return None

        with open(jj(folder, "inferred_topics.pickle"), 'rb') as f:
            pickled_data: types.SimpleNamespace = pickle.load(f)

        data: InferredTopicsData = InferredTopicsData(
            document_index=pickled_data.document_index
            if hasattr(pickled_data, 'document_index')
            else pickled_data.documents,
            dictionary=pickled_data.dictionary,
            topic_token_weights=pickled_data.topic_token_weights,
            topic_token_overview=pickled_data.topic_token_overview,
            document_topic_weights=pickled_data.document_topic_weights,
        )
        return data

    @staticmethod
    def explode(source: str, target_folder: str = None, feather: bool = True) -> InferredTopicsData:
        """We don't like pickle"""

        if isinstance(source, InferredTopicsData):
            data = source
            assert target_folder is not None
        else:
            if not isfile(jj(source, "inferred_topics.pickle")):
                return None
            data: InferredTopicsData = InferredTopicsData.load(folder=source)

        data.store(target_folder=(target_folder or source), pickled=False, feather=feather)

        return data

    @staticmethod
    def store(data: InferredTopicsData, target_folder: str) -> None:

        filename: str = jj(target_folder, "inferred_topics.pickle")

        c_data = types.SimpleNamespace(
            documents=data.document_index,
            dictionary=data.dictionary,
            topic_token_weights=data.topic_token_weights,
            topic_token_overview=data.topic_token_overview,
            document_topic_weights=data.document_topic_weights,
        )
        with open(filename, 'wb') as f:
            pickle.dump(c_data, f, pickle.HIGHEST_PROTOCOL)
