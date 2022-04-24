from __future__ import annotations

import os
from dataclasses import dataclass
from os.path import join as jj
from typing import Any, Iterable, List, Literal, Mapping, Protocol

import pandas as pd
from loguru import logger

from penelope import corpus as pc
from penelope import topic_modelling as tm
from penelope.corpus.token2id import id2token2token2id
from penelope.utility import write_json
from penelope.vendor.gensim_api import corpora

from ..interfaces import ContentType, DocumentPayload, ITask
from ..tasks_mixin import DefaultResolveMixIn

# pylint: disable=too-many-instance-attributes


@dataclass
class TopicModelMixinProtocol(Protocol):
    target_folder: str = None
    target_name: str = None
    engine: str = None
    engine_args: dict = None
    store_corpus: bool = False
    store_compressed: bool = True
    prior: ITask = None
    document_index: pd.DataFrame = None

    def resolved_prior_out_content_type(self) -> ContentType:
        ...

    def instream_to_vectorized_corpus(self: TopicModelMixinProtocol, token2id: dict) -> pc.VectorizedCorpus:
        ...


class TopicModelMixin:

    # FIXME: Consolidate this function with StremVectorizer()
    def instream_to_vectorized_corpus(self: TopicModelMixinProtocol, token2id: dict) -> pc.VectorizedCorpus:
        """Create a sparse corpus of instream terms. Return `pc.VectorizedCorpus`.
        Note that terms not found in token2id are ignored. This will happen
        when a new corpus is predicted that have terms not found in the training corpus.
        """
        if self.resolved_prior_out_content_type() == ContentType.VECTORIZED_CORPUS:
            payload: DocumentPayload = next(self.prior.outstream())
            return payload.content

        corpus: pc.VectorizedCorpus = pc.CorpusVectorizer().fit_transform(
            corpus=self.prior.filename_content_stream(),
            already_tokenized=True,
            document_index=self.document_index.set_index('document_id', drop=False),
            vocabulary=token2id,
        )
        return corpus

    def predict(
        self: TopicModelMixinProtocol,
        *,
        inferred_model: tm.InferredModel,
        id2token: dict,
        corpus: corpora.Sparse2Corpus | pc.VectorizedCorpus,
        document_index: pd.DataFrame,
        target_folder: str,
        n_tokens: int,
        minimum_probability: float,
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
        if not isinstance(corpus, (pc.VectorizedCorpus, corpora.Sparse2Corpus)):
            # raise ValueError(f"predict: corpus type {type(corpus)} not supported in predict (use sparse instead)")
            corpus = self.instream_to_vectorized_corpus(token2id=id2token2token2id(id2token))

        if isinstance(corpus, pc.VectorizedCorpus):
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
        inferred_model.store_options(target_folder)
        topics_data.store(target_folder)
        return topics_data

    def train(self: TopicModelMixinProtocol, train_corpus: tm.TrainingCorpus) -> tm.InferredModel:

        inferred_model: tm.InferredModel = tm.train_model(
            train_corpus=train_corpus, method=self.engine, engine_args=self.engine_args
        )

        os.makedirs(self.target_subfolder, exist_ok=True)

        inferred_model.topic_model.save(jj(self.target_subfolder, 'gensim.model.gz'))

        inferred_model.store(
            folder=self.target_subfolder,
            store_compressed=self.store_compressed,
        )

        if self.store_corpus:
            train_corpus.store(self.target_subfolder)

        return inferred_model

    def ensure_target_path(self: TopicModelMixinProtocol):

        if self.target_folder is None or self.target_name is None:
            raise ValueError("expected target folder and target name, found None")

        os.makedirs(self.target_subfolder, exist_ok=True)

    @property
    def target_subfolder(self: TopicModelMixinProtocol) -> str:
        return jj(self.target_folder, self.target_name)


@dataclass
class ToTopicModel(TopicModelMixin, DefaultResolveMixIn, ITask):
    """Trains and/or predicts a topic model.

    Yields:
        Payload[Tuple[str,str]]: folder and tag to created model
    """

    """If not None, then existing training corpus will used"""
    train_corpus_folder: str = None

    """Target"""
    target_mode: Literal['train', 'predict', 'both'] = 'both'
    target_folder: str = None
    target_name: str = None

    """Training/prediction options"""
    trained_model_folder: str = None
    engine: str = "gensim_lda-multicore"
    engine_args: dict = None
    n_tokens: int = 200
    minimum_probability: float = 0.01

    """If true, then training corpus will bes tored"""
    store_corpus: bool = False
    store_compressed: bool = True

    def __post_init__(self):

        self.in_content_type = [ContentType.TOKENS, ContentType.VECTORIZED_CORPUS]
        self.out_content_type = ContentType.TOPIC_MODEL

    def setup(self) -> ITask:
        super().setup()
        self.pipeline.put("topic_modeling_opts", self.engine_args)
        return self

    def instream_to_corpus(self, id2token: Mapping[int, str] | None) -> tm.TrainingCorpus:

        content_type: ContentType = self.resolved_prior_out_content_type()

        if self.train_corpus_folder:

            if tm.TrainingCorpus.exists(self.train_corpus_folder):
                logger.info(
                    f"using existing corpus in folder {self.train_corpus_folder} for target mode {self.target_mode}"
                )
                corpus: tm.TrainingCorpus = tm.TrainingCorpus.load(self.train_corpus_folder)
                return corpus

            tags: List[str] = pc.VectorizedCorpus.find_tags(self.train_corpus_folder)

            if len(tags) == 0:
                raise ValueError(f"no train or predict input corpus found in {self.train_corpus_folder}")

            if len(tags) > 1:
                raise ValueError(f"multiple corpus found in folder {self.train_corpus_folder}")

            logger.info(
                f"using corpus tagged {tags[0]} in folder {self.train_corpus_folder} for target mode {self.target_mode}"
            )
            vectorized_corpus: pc.VectorizedCorpus = pc.VectorizedCorpus.load(
                folder=self.train_corpus_folder, tag=tags[0]
            )
            corpus: tm.TrainingCorpus = tm.TrainingCorpus(corpus=vectorized_corpus)
            return corpus

        if content_type == ContentType.VECTORIZED_CORPUS:

            logger.info("creating sparse corpus out of input stream...")

            payload: DocumentPayload = next(self.prior.outstream())
            vectorized_corpus: pc.VectorizedCorpus = payload.content
            vectorize_opts: pc.VectorizeOpts = payload.recall('vectorize_opts')

            if id2token is not None:
                """We must consolidate the vocabularies"""
                logger.info("translating vocabulary to training model's vocabulary...")

                vectorized_corpus.translate_to_vocab(id2token, inplace=True)

            corpus: tm.TrainingCorpus = tm.TrainingCorpus(
                corpus=vectorized_corpus,
                corpus_options={},
                vectorizer_args={} if vectorize_opts is None else vectorize_opts.props,
            )
            logger.info("training corpus created!")

            return corpus

        if content_type == ContentType.TOKENS:
            token2id: pc.Token2Id = (
                pc.Token2Id(pc.id2token2token2id(id2token)) if id2token is not None else self.pipeline.payload.token2id
            )
            corpus: tm.TrainingCorpus = tm.TrainingCorpus(
                corpus=self.prior.filename_content_stream(),
                document_index=self.document_index,
                token2id=token2id,
                corpus_options={},
            )
            return corpus

        raise ValueError("unable to resolve input corpus")

    def process_stream(self) -> Iterable[DocumentPayload]:

        self.input_type_guard(self.resolved_prior_out_content_type())

        self.ensure_target_path()

        """
        If `predict` we only allow content_type == ContentType.TOKENS, and we must use
        the same vocabulary as was used in the training (at least for LdaMulticore)
        when translating the stream to a sparse corpus.
        """

        # inferred_model: tm.InferredModel = (
        #     tm.InferredModel.load(self.trained_model_folder, lazy=False)
        #     if self.target_mode == 'predict'
        #     else self.train(train_corpus)
        # )

        inferred_model: tm.InferredModel = None
        predict_corpus: Any = None

        if self.target_mode in ['both', 'train']:

            train_corpus: tm.TrainingCorpus = self.instream_to_corpus(id2token=None)
            predict_corpus = train_corpus.corpus

            inferred_model: tm.InferredModel = self.train(train_corpus)

        if self.target_mode in ['both', 'predict']:

            if self.target_mode == 'predict':

                logger.info("loading topic model...")
                inferred_model = tm.InferredModel.load(self.trained_model_folder, lazy=False)

                logger.info("preparing corpus for prediction...")
                predict_corpus = self.instream_to_corpus(id2token=inferred_model.id2token).corpus

            logger.info("prediction started")
            _ = self.predict(
                inferred_model=inferred_model,
                corpus=predict_corpus,
                id2token=inferred_model.id2token,
                document_index=self.document_index,
                target_folder=self.target_subfolder,
                n_tokens=self.n_tokens,
                minimum_probability=self.minimum_probability,
            )

            if self.pipeline and self.pipeline.config:
                self.pipeline.config.dump(jj(self.target_subfolder, "corpus.yml"))

            logger.info("prediction done")

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
    minimum_probability: float = 0.01

    @property
    def model_subfolder(self) -> str:
        return jj(self.model_folder, self.model_name)

    def __post_init__(self):

        self.in_content_type = [ContentType.TOKENS, ContentType.VECTORIZED_CORPUS]
        self.out_content_type = ContentType.TOPIC_MODEL

    def process_stream(self) -> Iterable[DocumentPayload]:

        self.ensure_target_path()

        trained_model: tm.InferredModel = tm.InferredModel.load(folder=self.model_subfolder, lazy=True)
        topics_data: tm.InferredTopicsData = tm.InferredTopicsData.load(folder=self.model_subfolder)

        corpus: pc.VectorizedCorpus = self.instream_to_vectorized_corpus(token2id=topics_data.term2id)

        self.pipeline.payload.token2id = topics_data.token2id

        _ = self.predict(
            inferred_model=trained_model,
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
            data=trained_model.options,
            default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>",
        )

        # FIXME: Fix ambiguity between self.target_folder/self.target_name and engine_args['work_folder]
        # self.target_name is never used??
        payload: DocumentPayload = DocumentPayload(
            ContentType.TOPIC_MODEL,
            content=dict(
                target_name=self.target_name,
                target_folder=self.target_folder,
            ),
        )

        yield payload
