import os
from dataclasses import dataclass
from typing import Iterable

from penelope import topic_modelling
from tqdm.auto import tqdm

from ..interfaces import ContentType, DocumentPayload, ITask


class ReiterableTerms:
    def __init__(self, outstream):
        self.outstream = outstream

    def __iter__(self):
        return (p.content for p in self.outstream())


@dataclass
class ToTopicModel(ITask):
    """Computes topic model.

    Iterable[DocumentPayload] => ComputeResult
    """

    corpus_filename: str = None
    corpus_folder: str = None
    target_folder: str = None
    target_name: str = None
    engine: str = "gensim_lda-multicore"
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

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        return None

    def process_stream(self) -> Iterable[DocumentPayload]:

        if self.target_folder is None or self.target_name is None:
            raise ValueError("expceted target folder and target name, found None")

        target_folder: str = os.path.join(self.target_folder, self.target_name)
        os.makedirs(target_folder, exist_ok=True)

        # content_type: ContentType = self.prior.out_content_type if self.prior else None
        # terms, document_index = self.get_tokens_stream(content_type)

        terms = tqdm(ReiterableTerms(self.prior.outstream), total=len(self.document_index))

        train_corpus: topic_modelling.TrainingCorpus = topic_modelling.TrainingCorpus(
            terms=terms,
            document_index=self.document_index,
            corpus_options={},
        )

        inferred_model: topic_modelling.InferredModel = topic_modelling.infer_model(
            train_corpus=train_corpus, method=self.engine, engine_args=self.engine_args
        )

        inferred_model.topic_model.save(os.path.join(target_folder, 'gensim.model.gz'))

        topic_modelling.store_model(
            inferred_model, target_folder, store_corpus=self.store_corpus, store_compressed=self.store_compressed
        )

        inferred_topics: topic_modelling.InferredTopicsData = topic_modelling.compile_inferred_topics_data(
            inferred_model.topic_model, train_corpus.corpus, train_corpus.id2word, train_corpus.document_index
        )

        inferred_topics.store(target_folder)

        payload: DocumentPayload = DocumentPayload(
            ContentType.TOPIC_MODEL,
            content=dict(target_name=self.target_name, target_folder=self.target_folder),
        )

        yield payload

    # # FIXME: Finalize implementation of function if multipe in_content_type should be allwoed
    # def get_tokens_stream(self, content_type: ContentType) -> Tuple[Iterable[Iterable[str]], DocumentIndex]:

    #     # TODO Implement these content types? Or defer to previous tasks in chain? (simpler)
    #     # if content_type is None:
    #     #     corpus: TokenizedCorpus = TokenizedCorpus(
    #     #         reader=TextTokenizer(
    #     #             source=self.corpus_filename,
    #     #             reader_opts=self.reader_opts or self.pipeline.config.text_reader_opts,
    #     #         ),
    #     #     )

    #     #     return corpus.terms, self.document_index or corpus.document_index

    #     # if content_type == ContentType.TEXT:
    #     #     corpus: TokenizedCorpus = TokenizedCorpus(
    #     #         reader=TextTokenizer(
    #     #             source=((p.filename, p.content) for p in self.prior.outstream()),
    #     #         ),
    #     #     )
    #     #     return corpus.terms, self.document_index or corpus.document_index

    #     if content_type == ContentType.TOKENS:
    #         return (p.content for p in self.prior.outstream()), self.document_index

    #     raise ValueError("content type not valid")
