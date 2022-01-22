from typing import Callable, Iterable

import numpy as np
import pandas as pd

from penelope import corpus as pc

from .. import interfaces


class StreamVectorizer:
    def __init__(
        self,
        token2id: pc.Token2Id,
        document_index: pd.DataFrame,
        vectorize_opts: pc.VectorizeOpts,
        tagged_column: str,
        tokenizer: Callable[[str], Iterable[str]] = None,
    ):
        self.token2id: pc.Token2Id = token2id
        self.document_index: pd.DataFrame = document_index
        self.vectorize_opts: pc.VectorizeOpts = vectorize_opts
        self.tagged_column: str = tagged_column
        self.tokenizer: Callable[[str], Iterable[str]] = tokenizer

    def from_token_id_stream(self, stream: Iterable[interfaces.DocumentPayload]) -> pc.VectorizedCorpus:
        corpus: pc.VectorizedCorpus = pc.VectorizedCorpus.from_token_id_stream(
            stream=stream,
            token2id=self.token2id,
            document_index=self.document_index,
            min_tf=self.vectorize_opts.min_tf,
            max_tokens=self.vectorize_opts.max_tokens,
        )
        return corpus

    def vectorize_stream(self, content_type: interfaces.ContentType, payloads: Iterable[interfaces.DocumentPayload]):

        name2id: dict = self.document_index['document_id'].to_dict().get

        if content_type == interfaces.ContentType.TOKENS:
            if self.token2id is not None:
                fg: dict = self.token2id.data.get
                tokens2series = lambda tokens: pd.Series([fg(t) for t in tokens], dtype=np.int32)
                stream = [(name2id(p.document_name), tokens2series(p.content)) for p in payloads]
                vectorized_corpus: pc.VectorizedCorpus = self.from_token_id_stream(stream)
            else:
                stream = ((p.document_name, p.content) for p in payloads)
                vectorized_corpus: pc.VectorizedCorpus = pc.CorpusVectorizer().fit_transform_(
                    stream,
                    document_index=self.document_index,
                    vectorize_opts=self.vectorize_opts.update(already_tokenized=True),
                )
        elif content_type == interfaces.ContentType.TOKEN_IDS:
            stream = [(name2id(p.document_name), pd.Series(p.content, dtype=np.int32)) for p in payloads]
            vectorized_corpus: pc.VectorizedCorpus = self.from_token_id_stream(stream)

        elif content_type == interfaces.ContentType.TAGGED_ID_FRAME:
            if self.tagged_column is None:
                raise ValueError("tagged column name in source must be specified!")

            tagged_column: str = self.tagged_column
            stream = ((name2id(p.document_name), p.content[tagged_column]) for p in payloads)
            vectorized_corpus: pc.VectorizedCorpus = self.from_token_id_stream(stream)

        elif content_type == interfaces.ContentType.TEXT:
            tokenizer = self.tokenizer or pc.default_tokenizer
            stream = ((p.document_name, tokenizer(p.content)) for p in payloads)
            vectorized_corpus: pc.VectorizedCorpus = pc.CorpusVectorizer().fit_transform_(
                stream,
                document_index=self.document_index,
                vectorize_opts=self.vectorize_opts.update(already_tokenized=True),
            )

        elif content_type == interfaces.ContentType.TAGGED_FRAME:
            raise NotImplementedError("Obselete: use TaggedFrameToTokens")

        else:
            raise ValueError(f"not supported: {content_type}")
        return vectorized_corpus
