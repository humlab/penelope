from __future__ import annotations

import os
from dataclasses import dataclass
from os.path import join as jj
from typing import Iterable

import pandas as pd
from gensim.matutils import Sparse2Corpus
from penelope import topic_modelling as tm
from penelope.corpus import CorpusVectorizer, VectorizedCorpus
from penelope.topic_modelling.engine_gensim.options import EngineKey
from penelope.utility.file_utility import write_json

from ..interfaces import ContentType, DocumentPayload, ITask
from ..tasks_mixin import DefaultResolveMixIn


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

        if isinstance(corpus, VectorizedCorpus):
            """Make sure we use corpus' own data"""
            document_index = corpus.document_index
            id2token = corpus.id2token

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

        inferred_model.store(
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

    def __post_init__(self):

        self.in_content_type = ContentType.TOKENS
        self.out_content_type = ContentType.TOPIC_MODEL

    def setup(self) -> ITask:
        super().setup()
        self.pipeline.put("topic_modeling_opts", self.engine_args)
        return self

    def instream_to_corpus(self) -> tm.TrainingCorpus:

        corpus: tm.TrainingCorpus = tm.TrainingCorpus(
            terms=self.prior.content_stream(), document_index=self.document_index, corpus_options={}
        )
        return corpus

    def process_stream(self) -> Iterable[DocumentPayload]:

        self.input_type_guard(self.prior.out_content_type)

        self.ensure_target_path()

        train_corpus: tm.TrainingCorpus = self.instream_to_corpus()

        inferred_model: tm.InferredModel = self.train(train_corpus)

        # FIXME: MALLET PREDICTION!
        _ = self.predict(
            inferred_model=inferred_model,
            corpus=train_corpus.corpus,
            id2token=train_corpus.id2token,
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

    @property
    def target_subfolder(self) -> str:
        return jj(self.target_folder, self.target_name)

    def __post_init__(self):

        self.in_content_type = [ContentType.TOKENS, ContentType.VECTORIZED_CORPUS]
        self.out_content_type = ContentType.TOPIC_MODEL

    def instream_to_vectorized_corpus(self, token2id: dict) -> VectorizedCorpus:
        """Create a sparse corpus of instream terms. Return `VectorizedCorpus`.
        Note that terms not found in token2id are ignored. This will happen
        when a new corpus is predicted that have terms not found in the training corpus.
        """
        if self.prior.out_content_type == ContentType.VECTORIZED_CORPUS:
            payload: DocumentPayload = next(self.prior.outstream())
            return payload.content

        corpus: VectorizedCorpus = CorpusVectorizer().fit_transform(
            corpus=self.prior.filename_content_stream(),
            already_tokenized=True,
            document_index=self.document_index.set_index('document_id', drop=False),
            vocabulary=token2id,
        )
        return corpus

    def process_stream(self) -> Iterable[DocumentPayload]:

        self.ensure_target_path()

        inferred_model: tm.InferredModel = tm.InferredModel.load(folder=self.model_subfolder, lazy=True)
        topics_data: tm.InferredTopicsData = tm.InferredTopicsData.load(folder=self.model_subfolder)

        corpus: VectorizedCorpus = self.instream_to_vectorized_corpus(token2id=topics_data.term2id)

        self.pipeline.payload.token2id = topics_data.token2id

        _ = self.predict(
            inferred_model=inferred_model,
            corpus=corpus,
            id2token=corpus.id2token,
            document_index=corpus.document_index,
            topic_token_weights=topics_data.topic_token_weights,
            topic_token_overview=topics_data.topic_token_overview,
            n_tokens=self.n_tokens,
            minimum_probability=self.minimum_probability,
            target_folder=self.target_subfolder,
        )

        corpus.dump(tag='predict', folder=self.target_subfolder, mode='files')

        write_json(
            path=jj(self.target_subfolder, "model_options.json"),
            data=inferred_model.options,
            default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>",
        )

        payload: DocumentPayload = DocumentPayload(
            ContentType.TOPIC_MODEL,
            content=dict(
                target_name=self.target_name,
                target_folder=self.target_folder,
            ),
        )

        yield payload
