from __future__ import annotations

import os
import pickle
import types
from contextlib import suppress
from functools import cached_property
from os.path import isfile
from os.path import join as jj
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Self, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from penelope import corpus as pc
from penelope import utility as pu
from penelope.utility.filename_fields import FilenameFieldSpecs

from . import token as tt
from .document import DocumentTopicsCalculator

if TYPE_CHECKING:
    from penelope.pipeline import CorpusConfig


CSV_OPTS: dict = dict(sep='\t', header=0, index_col=0, na_filter=False)

# pylint: disable=too-many-public-methods,access-member-before-definition,attribute-defined-outside-init,no-member


def smart_load(
    filename: str, *, missing_ok: bool = False, feather_pipe: Callable[[pd.DataFrame], pd.DataFrame] = None, **kwargs
) -> pd.DataFrame:
    feather_filename: str = pu.replace_extension(filename, "feather")
    if isfile(feather_filename):
        data: pd.DataFrame = pd.read_feather(feather_filename)
        if feather_pipe is not None:
            data = data.pipe(feather_pipe, **kwargs)
    elif isfile(filename):
        data = pd.read_csv(filename, **CSV_OPTS)
    else:
        if missing_ok:
            return None
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


DTYPES: dict = {
    'year': np.int16,
    'n_tokens': np.int32,
    'n_raw_tokens': np.int32,
    'document_id': np.int32,
    'topic_id': np.int16,
    'token_id': np.int32,
    'weight': np.float32,
    'label': str,
}


class SlimItMixIn:
    def remove_series_nan(self, series: pd.Series) -> pd.Series:
        if series.isnull().values.any():
            logger.warning(f"slim_types: {sum(series.isnull())} NaN found in {series.name}")
            return series.fillna(0)
        return series

    def slim_series(self, series: pd.Series, dtype: np.dtype) -> pd.Series:
        try:
            return series.astype(dtype)
        except (ValueError, pd.errors.IntCastingNaNError):
            if np.issubdtype(dtype, np.integer):
                return pd.to_numeric(series, errors='coerce').fillna(0).astype(dtype)
            raise

    def slim_dataframe(self, df: pd.DataFrame, dtypes: dict) -> pd.DataFrame:
        if df is not None:
            columns: set[str] = set(dtypes.keys()).intersection(set(df.columns))
            for column in columns:
                df[column] = self.slim_series(df[column], dtypes[column])
        return df

    def slim_types(self) -> Self:
        self.topic_token_weights['token_id'] = self.remove_series_nan(self.topic_token_weights.token_id)

        pos_dtypes: dict = {x: np.int32 for x in pu.PD_PoS_tag_groups.index.to_list()}

        self.slim_dataframe(self.document_index, dtypes={**DTYPES, **pos_dtypes})  # type: ignore
        self.slim_dataframe(self.topic_token_weights, dtypes=DTYPES)  # type: ignore
        self.slim_dataframe(self.document_topic_weights, dtypes=DTYPES)  # type: ignore
        self.slim_dataframe(self.topic_token_overview, dtypes=DTYPES)  # type: ignore

        return self

    def slimmer(self) -> Self:
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
        topic_diagnostics: pd.DataFrame | None,
        token_diagnostics: pd.DataFrame | None,
        corpus_config: CorpusConfig | None = None,
        overload_data: bool = True,
    ):
        """Container for predicted topics data."""
        super().__init__()

        self.dictionary: pd.DataFrame = dictionary
        self.document_index: pd.DataFrame = document_index
        self.topic_token_weights: pd.DataFrame = topic_token_weights
        self.document_topic_weights: pd.DataFrame = (
            pc.DocumentIndexHelper(document_index).overload(document_topic_weights, 'year')
            if overload_data
            else document_topic_weights
        )

        self.topic_token_overview: pd.DataFrame = topic_token_overview
        self.topic_diagnostics: pd.DataFrame | None = topic_diagnostics
        self.token_diagnostics: pd.DataFrame | None = token_diagnostics
        self.calculator = DocumentTopicsCalculator(self)
        self.corpus_config: CorpusConfig = corpus_config

        if overload_data and 'token' not in self.topic_token_weights.columns:
            self.topic_token_weights['token'] = self.topic_token_weights['token_id'].apply(self.id2term.get)

    def copy(self) -> InferredTopicsData:
        return InferredTopicsData(
            dictionary=self.dictionary.copy(),
            document_index=self.document_index.copy(),
            topic_token_weights=self.topic_token_weights.copy(),
            document_topic_weights=self.document_topic_weights.copy(),
            topic_token_overview=self.topic_token_overview.copy(),
            topic_diagnostics=self.topic_diagnostics.copy() if self.topic_diagnostics is not None else None,
            token_diagnostics=self.token_diagnostics.copy() if self.token_diagnostics is not None else None,
            corpus_config=self.corpus_config,
            overload_data=False,
        )

    @property
    def document_index_proper(self) -> pd.DataFrame:
        return (
            self.document_index.assign(document_id=self.document_index.index)
            .set_index('document_name', drop=False)
            .rename_axis('')
        )

    @property
    def num_topics(self) -> int:
        return int(self.topic_token_overview.index.max()) + 1

    @property
    def n_topics(self) -> int:
        return self.num_topics

    @property
    def timespan(self) -> Tuple[int | None, int | None]:
        return self.year_period

    def startspan(self, n: int) -> tuple[int, int, int]:
        return (self.year_period[0], min(self.year_period[1], self.year_period[0] + n))

    def stopspan(self, n: int) -> tuple[int, int, int]:
        return (max(self.year_period[0], self.year_period[1] - n), self.year_period[1])

    @property
    def year_period(self) -> tuple[int | None, int | None]:
        """Returns documents `year` interval (if exists)"""
        if 'year' not in self.document_topic_weights.columns:
            return (None, None)
        return (self.document_topic_weights.year.min(), self.document_topic_weights.year.max())

    @property
    def topic_ids(self) -> List[int]:
        """Returns unique topic ids"""
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
        """Stores topics data in `target_folder` either as pickled file or individual zipped files"""
        os.makedirs(target_folder, exist_ok=True)
        if pickled:
            PickleUtility.store(self, target_folder=target_folder)
        else:
            self._store_csv(target_folder)
            if feather:
                self._store_feather(target_folder)

        if self.corpus_config is not None:
            self.corpus_config.dump(jj(target_folder, "corpus.yml"))
            if self.corpus_config.corpus_version:
                (Path(target_folder) / 'version').write_text(self.corpus_config.corpus_version)

    def _store_csv(self, target_folder: str) -> None:
        data: list[tuple[pd.DataFrame | None, str]] = [
            (self.document_index.rename_axis(''), 'documents.csv'),
            (self.dictionary, 'dictionary.csv'),
            (self.topic_token_weights, 'topic_token_weights.csv'),
            (self.topic_token_overview, 'topic_token_overview.csv'),
            (self.document_topic_weights, 'document_topic_weights.csv'),
            (self.topic_diagnostics, 'topic_diagnostics.csv'),
            (self.token_diagnostics, 'token_diagnostics.csv'),
        ]

        for df, name in data:
            if df is None:
                continue
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

        if self.topic_diagnostics is not None:
            self.topic_diagnostics.reset_index().to_feather(jj(target_folder, "topic_diagnostics.feather"))
        if self.token_diagnostics is not None:
            self.token_diagnostics.reset_index(drop=True).to_feather(jj(target_folder, "token_diagnostics.feather"))

    @staticmethod
    def load(*, folder: str, slim: bool = False, verbose: bool = False):
        """Loads previously stored aggregate"""

        corpus_config: CorpusConfig | None = InferredTopicsData.load_corpus_config(folder)

        if corpus_config is None:
            raise FileNotFoundError(f"No CorpusConfig found in {folder}")

        filename_fields: FilenameFieldSpecs = corpus_config.text_reader_opts.filename_fields

        document_index: pd.DataFrame = (
            pd.read_feather(jj(folder, "documents.feather")).rename_axis('document_id')
            if isfile(jj(folder, "documents.feather"))
            else pc.load_document_index(
                jj(folder, 'documents.zip'), filename_fields=filename_fields, **CSV_OPTS
            ).set_index('document_id', drop=True)
        )

        data: InferredTopicsData = InferredTopicsData(
            dictionary=smart_load(jj(folder, 'dictionary.zip'), feather_pipe=pu.set_index, columns='token_id'),
            document_index=document_index,
            topic_token_weights=smart_load(jj(folder, 'topic_token_weights.zip')),
            document_topic_weights=smart_load(jj(folder, 'document_topic_weights.zip')),
            topic_token_overview=smart_load(
                jj(folder, 'topic_token_overview.zip'), feather_pipe=pu.set_index, columns='topic_id'
            ),
            topic_diagnostics=smart_load(
                jj(folder, 'topic_diagnostics.zip'), missing_ok=True, feather_pipe=pu.set_index, columns='topic_id'
            ),
            token_diagnostics=smart_load(jj(folder, 'token_diagnostics.zip'), missing_ok=True),
            corpus_config=corpus_config,
        )

        # HACK: Handle renamed column:
        data.document_index = fix_renamed_columns(data.document_index)
        assert "year" in data.document_index.columns

        data.topic_token_overview = data.load_topic_token_label_overview(folder, **CSV_OPTS)

        data.slim_types()
        if slim:
            data.slimmer()
        if verbose:
            data.log_usage(total=True)
        return data

    @staticmethod
    def load_corpus_config(folder: str) -> CorpusConfig | None:
        """Load CorpusConfig if exists"""
        corpus_configs: list[CorpusConfig] = pu.create_class("penelope.pipeline.CorpusConfig").find_all(  # type: ignore
            folder=folder, set_folder=True
        )

        if len(corpus_configs) > 0:
            return corpus_configs[0]

        logger.warning(f'No CorpusConfig found in {folder} (may affect certain operations)')
        return None

    @staticmethod
    def topic_labels_filename(private: bool = False) -> str:
        username: str = os.environ.get("JUPYTERHUB_USER", "")
        if private and username:
            return f"topic_token_overview_label-{username}.csv"
        return "topic_token_overview_label.csv"

    def load_topic_token_label_overview(self, folder: str, private: bool = False, **csv_opts: dict) -> pd.DataFrame:
        """Loads labeled topic_token_overview if exists, otherwise add id->'id' mapping as labels."""
        tto: pd.DataFrame = self.topic_token_overview
        filename: str = jj(folder, self.topic_labels_filename(private))
        if isfile(filename):
            labeled_tto: pd.DataFrame = pd.read_csv(filename, **csv_opts)  # type: ignore
            if self.is_satisfied_topic_token_overview(labeled_tto):
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
        dictionary: pd.DataFrame = smart_load(
            jj(folder, 'dictionary.zip'), feather_pipe=pu.set_index, columns='token_id'
        )
        token2id: pc.Token2Id = pc.Token2Id(data={t: i for (t, i) in zip(dictionary.token, dictionary.index)})
        return token2id

    @cached_property
    def topic_labels(self) -> dict:
        if 'label' not in self.topic_token_overview.columns:
            return {}
        return self.topic_token_overview['label'].to_dict()

    def get_topics_overview_with_score(self, n_tokens: int = 500):
        topics: pd.DataFrame = self.topic_token_overview
        topics['tokens'] = self.get_topic_titles(n_tokens=n_tokens)

        columns_to_show: list[str] = [column for column in ['tokens', 'alpha', 'coherence'] if column in topics.columns]

        topics = topics[columns_to_show]

        with suppress(BaseException):
            topic_proportions: pd.DataFrame = self.calculator.topic_proportions()
            if topic_proportions is not None:
                topics['score'] = topic_proportions

        if topics is None:
            raise ValueError("bug-check: No topic_token_overview in loaded model!")
        return topics

    def merge(self, cluster_mapping: dict[str, list[int]]) -> Self:

        def _validate_clusters(clusters: dict[str, list[int]]) -> dict[str, list[int]]:
            """Check if topic ids are valid"""
            for cluster_name, topic_ids in clusters.items():

                if any(t not in self.topic_token_overview.index for t in topic_ids):
                    raise ValueError(f'Invalid topic id found in cluster {cluster_name}')

                if len(topic_ids) < 2:
                    logger.warning(f'Ignoring cluster {cluster_name} since it has less than 2 topics')

            return {k: v for k, v in clusters.items() if len(v) > 1}

        def _merge_to_cluster(data: pd.DataFrame, recodes: dict[int, int]) -> pd.DataFrame:
            """Merge topic ids in data using recodes mapping"""
            columns: list[str] = data.columns.to_list()
            data['topic_id'] = data['topic_id'].replace(recodes)
            groupby_columns: list[str] = data.columns.difference(['weight']).tolist()
            data = data.groupby(groupby_columns).agg({'weight': 'sum'}).reset_index()
            return data[columns]

        cluster_mapping = _validate_clusters(cluster_mapping)

        """Merge each topic cluster identified by key, and topic ids (values) into a single topics"""
        for cluster_name, topic_ids in cluster_mapping.items():
            recode_map: dict[int, int] = {t: topic_ids[0] for t in topic_ids[1:]}

            self.document_topic_weights = _merge_to_cluster(self.document_topic_weights, recode_map)
            self.topic_token_weights = _merge_to_cluster(self.topic_token_weights, recode_map)

            # set column 'label' in dataframe self.topic_token_overview to cluster_name for first id in topic_id
            self.topic_token_overview.loc[topic_ids[0], 'label'] = cluster_name

        self.token_diagnostics = None
        self.topic_diagnostics = None

        return self

    def compress(self, n_types: int = 500) -> Self:
        """
        Compresses inferred topics by removing empty topics and recoding topic ids.
        """

        tto: pd.DataFrame = self.topic_token_overview
        ttw: pd.DataFrame = self.topic_token_weights
        dtw: pd.DataFrame = self.document_topic_weights

        keep_topics: set[int] = set(dtw.topic_id).union(ttw.topic_id)
        empty_topics: set[int] = set(tto.index).difference(keep_topics)

        if len(empty_topics) == 0:
            return self

        recoded_topic_ids: dict[int, int] = {t_id: i for i, t_id in enumerate(sorted(keep_topics))}
        labels: dict[int, str] = self.topic_token_overview['label'].to_dict()
        recoded_labels: dict[int, str] = {
            recoded_topic_ids[t_id]: label for t_id, label in labels.items() if t_id in recoded_topic_ids
        }

        dtw['topic_id'] = dtw['topic_id'].replace(recoded_topic_ids)
        ttw['topic_id'] = ttw['topic_id'].replace(recoded_topic_ids)

        self.document_topic_weights = dtw
        self.topic_token_weights = ttw

        self.topic_token_overview = compute_topic_token_overview(ttw, self.id2token, n_types)
        self.topic_token_overview['label'] = self.topic_token_overview.index.map(recoded_labels)

        return self


def compute_topic_token_overview(
    topic_type_weights: pd.DataFrame, id2type: dict[int, str], n_types: int = 500
) -> pd.DataFrame:
    """
    Group by topic_id and concatenate n_tokens words within group sorted by weight descending.
    There must be a better way of doing this...
    """
    overview: pd.DataFrame = (
        topic_type_weights.groupby('topic_id')
        .apply(lambda x: sorted(list(zip(x["token_id"], x["weight"])), key=lambda z: z[1], reverse=True))
        .apply(lambda x: [z[0] for z in x][:n_types])
        .reset_index()
    )
    overview.columns = ['topic_id', 'token_ids']

    overview['tokens'] = overview['token_ids'].apply(  # type: ignore
        lambda token_ids: ' '.join([id2type.get(token_id) for token_id in token_ids])  # type: ignore
    )
    overview['alpha'] = 0.0
    overview = overview.set_index('topic_id')

    return overview


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
    @staticmethod
    def load(folder: str) -> InferredTopicsData | None:
        if not isfile(jj(folder, "inferred_topics.pickle")):
            return None

        with open(jj(folder, "inferred_topics.pickle"), 'rb') as f:
            pickled_data: types.SimpleNamespace = pickle.load(f)

        data: InferredTopicsData = InferredTopicsData(
            document_index=(
                pickled_data.document_index if hasattr(pickled_data, 'document_index') else pickled_data.documents
            ),
            dictionary=pickled_data.dictionary,
            topic_token_weights=pickled_data.topic_token_weights,
            topic_token_overview=pickled_data.topic_token_overview,
            document_topic_weights=pickled_data.document_topic_weights,
            topic_diagnostics=None,
            token_diagnostics=None,
            corpus_config=pickled_data.corpus_config if hasattr(pickled_data, 'corpus_config') else None,
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
            corpus_config=data.corpus_config,
        )
        with open(filename, 'wb') as f:
            pickle.dump(c_data, f, pickle.HIGHEST_PROTOCOL)
