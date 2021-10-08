from __future__ import annotations

import os
from dataclasses import dataclass
from os.path import join as jj
from typing import Iterable

import pandas as pd
from gensim.matutils import Sparse2Corpus
from penelope import topic_modelling as tm
from penelope.corpus.dtm.corpus import VectorizedCorpus
from penelope.topic_modelling.engine_gensim.options import EngineKey
from tqdm.auto import tqdm

from ..interfaces import ContentType, DocumentPayload, ITask
from ..tasks_mixin import DefaultResolveMixIn


class ReiterableTerms:
    def __init__(self, outstream):
        self.outstream = outstream

    def __iter__(self):
        return (p.content for p in self.outstream())


class TopicModelMixin:
    def predict(
        self,
        *,
        inferred_model: tm.InferredModel,
        id2token: dict,
        corpus: Sparse2Corpus | VectorizedCorpus,
        document_index: pd.DataFrame,
        target_folder: str,
        n_tokens: int = 200,
        minimum_probability: float = 0.001,
        **kwargs,
    ) -> tm.InferredTopicsData:
        """[summary]

        Args:
            inferred_model (tm.InferredModel): [description]
            id2token (dict): [description]
            corpus (Sparse2Corpus): [description]
            document_index (pd.DataFrame): [description]
            target_folder (str): [description]
            topics_data (tm.InferredTopicsData, optional): If set, pick data from thi. Defaults to None.
            n_tokens (int, optional): [description]. Defaults to 200.
            minimum_probability (float, optional): [description]. Defaults to 0.001.

        Raises:
            ValueError: [description]

        Returns:
            tm.InferredTopicsData: [description]
        """
        if not isinstance(corpus, (VectorizedCorpus, Sparse2Corpus)):
            raise ValueError(f"predict: corpus type {type(corpus)} not supported in predict")

        topics_data: tm.InferredTopicsData = tm.predict_topics(
            inferred_model.topic_model,
            corpus=corpus,
            id2token=id2token,
            document_index=document_index,
            n_tokens=n_tokens,
            minimum_probability=minimum_probability,
            **kwargs,
        )

        topics_data.store(target_folder)
        return topics_data

    def train(self, train_corpus: tm.TrainingCorpus) -> tm.InferredModel:

        inferred_model: tm.InferredModel = tm.train_model(
            train_corpus=train_corpus, method=self.engine, engine_args=self.engine_args
        )

        inferred_model.topic_model.save(jj(self.target_subfolder, 'gensim.model.gz'))

        tm.store_model(
            inferred_model=inferred_model,
            folder=self.target_subfolder,
            store_corpus=self.store_corpus,
            store_compressed=self.store_compressed,
        )

        return inferred_model

    def ensure_target_path(self):

        if self.target_folder is None or self.target_name is None:
            raise ValueError("expected target folder and target name, found None")

        os.makedirs(jj(self.target_folder, self.target_name), exist_ok=True)

    @property
    def target_subfolder(self) -> str:
        return jj(self.target_folder, self.target_name)


@dataclass
class ToTopicModel(TopicModelMixin, DefaultResolveMixIn, ITask):
    """Computes topic model.

    Iterable[DocumentPayload] => ComputeResult
    """

    corpus_filename: str = None
    corpus_folder: str = None
    target_folder: str = None
    target_name: str = None
    engine: EngineKey = "gensim_lda-multicore"
    engine_args: dict = None
    store_corpus: bool = False
    store_compressed: bool = True

    # reader_opts: TextReaderOpts = None

    def __post_init__(self):

        # self.in_content_type = [ContentType.NONE, ContentType.TOKENS, ContentType.TEXT]
        self.in_content_type = ContentType.TOKENS
        self.out_content_type = ContentType.TOPIC_MODEL

    def setup(self) -> ITask:
        super().setup()
        self.pipeline.put("topic_modeling_opts", self.engine_args)
        return self

    def instream_to_corpus(self) -> tm.TrainingCorpus:

        terms = tqdm(ReiterableTerms(self.prior.outstream), total=len(self.document_index))
        corpus: tm.TrainingCorpus = tm.TrainingCorpus(
            terms=terms, document_index=self.document_index, corpus_options={}
        )
        return corpus

    def process_stream(self) -> Iterable[DocumentPayload]:

        self.input_type_guard(self.prior.out_content_type)

        self.ensure_target_path()

        train_corpus: tm.TrainingCorpus = self.instream_to_corpus()

        inferred_model: tm.InferredModel = self.train(train_corpus)

        _ = self.predict(
            inferred_model=inferred_model,
            corpus=train_corpus,
            id2token=None,
            document_index=self.document_index,
            target_folder=self.target_subfolder,
        )

        payload: DocumentPayload = DocumentPayload(
            ContentType.TOPIC_MODEL,
            content=dict(
                target_name=self.target_name,
                target_folder=self.target_folder,
            ),
        )

        yield payload


@dataclass
class PredictTopics(TopicModelMixin, DefaultResolveMixIn, ITask):
    """Predicts topics.

    Iterable[DocumentPayload] => ComputeResult
    """

    model_folder: str = None
    model_name: str = None
    target_folder: str = None
    target_name: str = None
    n_tokens: int = 200
    minimum_probability: float = 0.001

    @property
    def model_subfolder(self) -> str:
        return jj(self.model_folder, self.model_name)

    def __post_init__(self):

        self.in_content_type = ContentType.TOKENS
        self.out_content_type = ContentType.TOPIC_MODEL

    def instream_to_sparse_corpus(self) -> tm.TrainingCorpus:

        terms = tqdm(ReiterableTerms(self.prior.outstream), total=len(self.document_index))
        corpus: tm.TrainingCorpus = tm.TrainingCorpus(
            terms=terms, document_index=self.document_index, corpus_options={}
        )
        return corpus

    def process_stream(self) -> Iterable[DocumentPayload]:

        self.ensure_target_path()

        inferred_model: tm.InferredModel = tm.load_model(folder=self.model_subfolder, lazy=False)
        topics_data: tm.InferredTopicsData = tm.InferredTopicsData.load(folder=self.model_subfolder)

        corpus: VectorizedCorpus = self.instream_to_sparse_corpus()

        self.pipeline.payload.token2id = topics_data.token2id

        _ = self.predict(
            inferred_model=inferred_model,
            corpus=corpus,
            id2token=topics_data.id2term,
            document_index=self.document_index,
            topic_token_weights=topics_data.topic_token_weights,
            topic_token_overview=topics_data.topic_token_overview,
            n_tokens=self.n_tokens,
            minimum_probability=self.minimum_probability,
            target_folder=self.target_subfolder,
        )

        payload: DocumentPayload = DocumentPayload(
            ContentType.TOPIC_MODEL,
            content=dict(
                target_name=self.target_name,
                target_folder=self.target_folder,
            ),
        )

        yield payload
