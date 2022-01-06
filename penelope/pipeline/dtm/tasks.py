from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from penelope import corpus as pc

from .. import interfaces
from . import vectorizer


@dataclass
class ToDTM(interfaces.ITask):

    vectorize_opts: pc.VectorizeOpts = None
    tagged_column: Optional[str] = field(default=None)
    tokenizer: Callable[[str], Iterable[str]] = field(default=None)

    def __post_init__(self):
        self.in_content_type = [
            interfaces.ContentType.TEXT,
            interfaces.ContentType.TOKENS,
            interfaces.ContentType.TOKEN_IDS,
            interfaces.ContentType.TAGGED_FRAME,
            interfaces.ContentType.TAGGED_ID_FRAME,
        ]
        self.out_content_type = interfaces.ContentType.VECTORIZED_CORPUS

    def setup(self) -> interfaces.ITask:
        super().setup()
        self.pipeline.put("vectorize_opts", self.vectorize_opts)
        return self

    def process_stream(self) -> pc.VectorizedCorpus:

        content_type: interfaces.ContentType = self.resolved_prior_out_content_type()

        self.vectorize_opts.already_tokenized = True
        vectorized_corpus: pc.VectorizedCorpus = vectorizer.StreamVectorizer(
            token2id=self.pipeline.payload.token2id,
            document_index=self.document_index,
            vectorize_opts=self.vectorize_opts,
            tagged_column=self.tagged_column,
            tokenizer=self.tokenizer,
        ).vectorize_stream(
            content_type=content_type,
            payloads=self.create_instream(),
        )

        payload: interfaces.DocumentPayload = interfaces.DocumentPayload(
            content_type=interfaces.ContentType.VECTORIZED_CORPUS, content=vectorized_corpus
        )

        payload.remember(vectorize_opts=self.vectorize_opts)

        yield payload

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        return None
