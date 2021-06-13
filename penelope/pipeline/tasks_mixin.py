import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from penelope.corpus import Token2Id, TokensTransformer, TokensTransformOpts

from . import convert, interfaces


class DefaultResolveMixIn:
    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        return payload


class CountTaggedTokensMixIn:
    def register_token_counts(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        """Computes token counts from the tagged frame, and adds them to the document index"""
        try:

            if not isinstance(payload.content, pd.DataFrame):
                raise ValueError("setup error: register_token_counts only applicable for tagged frames")

            token_counts = convert.tagged_frame_to_token_counts(
                tagged_frame=payload.content,
                pos_schema=self.pipeline.payload.pos_schema,
                pos_column=self.pipeline.payload.get('pos_column'),
            )
            self.update_document_properties(payload, **token_counts)
            return payload
        except Exception as ex:
            logging.exception(ex)
            raise


@dataclass
class TransformTokensMixIn:

    transform_opts: Optional[TokensTransformOpts] = None
    transformer: Optional[TokensTransformer] = None

    def setup_transform(self) -> interfaces.ITask:

        if self.transform_opts is None:
            return self

        self.pipeline.put("transform_opts", self.transform_opts)

        if self.transformer is None:
            self.transformer = TokensTransformer(transform_opts=self.transform_opts)
        else:
            self.transformer.ingest(self.transform_opts)

        return self

    def transform(self, tokens: List[str]) -> List[str]:
        if self.transformer:
            return self.transformer.transform(tokens)
        return tokens


@dataclass
class VocabularyIngestMixIn:

    token2id: Token2Id = None
    ingest_tokens: bool = False

    def enter(self):
        super().enter()

        if self.ingest_tokens:
            if self.pipeline.payload.token2id is None:
                self.pipeline.payload.token2id = self.token2id or Token2Id()

        self.token2id = self.pipeline.payload.token2id

    def ingest(self, tokens: List[str]):

        if self.ingest_tokens:
            self.token2id.ingest(tokens)
