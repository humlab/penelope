from __future__ import annotations

import os
import pickle
import sys
import types
from functools import cached_property
from os import path as pp
from os.path import join as jj
from typing import List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from penelope import corpus as pc
from penelope import utility as pu

from . import token as tt
from .document import DocumentTopicsCalculator

CSV_OPTS: dict = dict(sep='\t', header=0, index_col=0, na_filter=False)


def smart_read(filename: str, **kwargs) -> pd.DataFrame:
    if pp.isfile(pu.replace_extension(filename, "feather")):
        return pd.read_feather(filename)
    return pd.read_csv(filename, **kwargs)


class InferredTopicsData(tt.TopicTokensMixIn):
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

    def info(self):
        for o_name in [k for k in self.__dict__ if not k.startswith("__")]:
            o_data = getattr(self, o_name)
            o_size = sys.getsizeof(o_data)
            print('{:>20s}: {:.4f} Mb {}'.format(o_name, o_size / (1024 * 1024), type(o_data)))

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
            # FIXME: deprecate pickled store
            filename: str = jj(target_folder, "inferred_topics.pickle")

            c_data = types.SimpleNamespace(
                documents=self.document_index,
                dictionary=self.dictionary,
                topic_token_weights=self.topic_token_weights,
                topic_token_overview=self.topic_token_overview,
                document_topic_weights=self.document_topic_weights,
            )
            with open(filename, 'wb') as f:
                pickle.dump(c_data, f, pickle.HIGHEST_PROTOCOL)

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
    def load(*, folder: str, filename_fields: pu.FilenameFieldSpecs = None, slim: bool = False, verbose: bool = False):
        """Loads previously stored aggregate"""
        data: InferredTopicsData = None

        if pp.isfile(jj(folder, 'documents.feather')):
            data: InferredTopicsData = InferredTopicsData(
                dictionary=pd.read_feather(jj(folder, "dictionary.feather")).set_index('token_id', drop=True),
                document_index=pd.read_feather(jj(folder, "documents.feather"))
                .set_index('document_name', drop=False)
                .rename_axis(''),
                topic_token_weights=pd.read_feather(jj(folder, "topic_token_weights.feather")),
                document_topic_weights=pd.read_feather(jj(folder, "document_topic_weights.feather")),
                topic_token_overview=pd.read_feather(jj(folder, "topic_token_overview.feather")).set_index(
                    'topic_id', drop=True
                ),
            )
        elif pp.isfile(jj(folder, 'documents.zip')):
            csv_opts: dict = dict(sep='\t', header=0, index_col=0, na_filter=False)
            data: InferredTopicsData = InferredTopicsData(
                dictionary=pd.read_csv(jj(folder, 'dictionary.zip'), **csv_opts),
                document_index=pc.load_document_index(
                    jj(folder, 'documents.zip'), filename_fields=filename_fields, **csv_opts
                ),
                topic_token_weights=pd.read_csv(jj(folder, 'topic_token_weights.zip'), **csv_opts),
                topic_token_overview=pd.read_csv(jj(folder, 'topic_token_overview.zip'), **csv_opts),
                document_topic_weights=pd.read_csv(jj(folder, 'document_topic_weights.zip'), **csv_opts),
            )

        elif pp.isfile(jj(folder, "inferred_topics.pickle")):

            with open(jj(folder, "inferred_topics.pickle"), 'rb') as f:
                pickled_data: types.SimpleNamespace = pickle.load(f)

            data: InferredTopicsData = InferredTopicsData(
                document_index=pickled_data.document_index
                if hasattr(pickled_data, 'document_index')
                else pickled_data.document,
                dictionary=pickled_data.dictionary,
                topic_token_weights=pickled_data.topic_token_weights,
                topic_token_overview=pickled_data.topic_token_overview,
                document_topic_weights=pickled_data.document_topic_weights,
            )

        else:
            raise FileNotFoundError(f"no model data found in {folder}")

        assert "year" in data.document_index.columns

        # HACK: Handle renamed column:
        data.document_index = fix_renamed_columns(data.document_index)

        # data.log_usage(total=True)
        data.slim_types()
        # data.log_usage(total=True)

        if slim:
            data.slimmer()

        if verbose:
            data.log_usage(total=True)

        return data

    @staticmethod
    def load_token2id(folder: str) -> pc.Token2Id:
        dictionary: pd.DataFrame = pd.read_csv(
            jj(folder, 'dictionary.zip'), sep='\t', header=0, index_col=0, na_filter=False
        )
        data: dict = {t: i for (t, i) in zip(dictionary.token, dictionary.index)}  # pylint: disable=no-member
        token2id: pc.Token2Id = pc.Token2Id(data=data)
        return token2id

    def slim_types(self) -> InferredTopicsData:

        """document_index"""
        self.document_index['year'] = self.document_index['year'].astype(np.int16)
        self.document_index['n_tokens'] = self.document_index['n_tokens'].astype(np.int32)
        self.document_index['n_raw_tokens'] = self.document_index['n_raw_tokens'].astype(np.int32)
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

        return self

    def slimmer(self) -> InferredTopicsData:

        """document_index"""
        remove_columns = set(pu.PD_PoS_tag_groups.index.to_list()) | {'filename', 'year2', 'number'}
        self.document_index.drop(columns=list(remove_columns.intersection(self.document_index.columns)), inplace=True)
        self.document_index.set_index('document_id', drop=True, inplace=True)

        """dictionary"""
        self.dictionary.drop(columns='dfs', inplace=True)

        """topic_token_weights"""
        if 'token_id' not in self.topic_token_weights.columns:
            self.topic_token_weights = self.topic_token_weights.reset_index().set_index('topic_id')

        if 'token' in self.topic_token_weights:
            self.topic_token_weights.drop(columns='token', inplace=True)

        # FIXME #149 Varför är weights `topic_token_weights` så stora tal???

        """document_topic_weights"""

        return self

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


def fix_renamed_columns(di: pd.DataFrame) -> pd.DataFrame:
    """Add count columns `n_tokens` and `n_rws_tokens" if missing and other/renamed column exists."""
    for missing in {"n_tokens", "n_raw_tokens"} - set(di.columns):
        for existing in {"n_terms", "n_tokens", "n_raw_tokens"}.intersection(di.columns):
            di[missing] = di[existing]
            break
    if 'n_terms' in di.columns:
        di = di.drop(columns='n_terms')
    return di


# a = {
#     'dictionary': {'Index': '1.5 MB', 'dfs': '1.5 MB', 'token': '15.0 MB'},
#     'document_index': {
#         'Adjective': '5.3 MB',
#         'Adverb': '5.3 MB',
#         'Conjunction': '5.3 MB',
#         'Delimiter': '5.3 MB',
#         'Index': '51.9 MB',
#         'Noun': '5.3 MB',
#         'Numeral': '5.3 MB',
#         'Other': '5.3 MB',
#         'Preposition': '5.3 MB',
#         'Pronoun': '5.3 MB',
#         'Verb': '5.3 MB',
#         'document_id': '2.7 MB',
#         'document_name': '51.9 MB',
#         'filename': '54.6 MB',
#         'n_raw_tokens': '5.3 MB',
#         'n_tokens': '5.3 MB',
#         'number': '16.0 MB',
#         'who': '52.4 MB',
#         'year': '1.3 MB',
#         'year2': '16.0 MB',
#     },
#     'document_topic_weights': {
#         'Index': '0.0 MB',
#         'document_id': '141.2 MB',
#         'topic_id': '141.2 MB',
#         'weight': '141.2 MB',
#         'year': '141.2 MB',
#     },
#     'topic_token_overview': {'Index': '0.0 MB', 'alpha': '0.0 MB', 'score': '0.0 MB', 'tokens': '0.1 MB'},
#     'topic_token_weights': {
#         'Index': '0.0 MB',
#         'token': '80.7 MB',
#         'token_id': '8.4 MB',
#         'topic_id': '8.4 MB',
#         'weight': '8.4 MB',
#     },
# }
