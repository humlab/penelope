import logging
from dataclasses import dataclass, field
from typing import List, Optional

from penelope.corpus import Token2Id, TokensTransformer, TokensTransformOpts

from . import convert
from .interfaces import DocumentPayload, ITask


class DefaultResolveMixIn:
    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        return payload


class CountTokensMixIn:
    def register_token_counts(self, payload: DocumentPayload) -> DocumentPayload:
        """Computes token counts from the tagged frame, and adds them to the document index"""
        try:
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

    def setup_transform(self) -> ITask:

        if self.transform_opts is None:
            return self

        self.pipeline.put("transform_opts", self.transform_opts)

        if self.transformer is None:
            self.transformer = TokensTransformer(transform_opts=self.transform_opts)

        self.transformer.ingest(self.transform_opts)

        return self

    def transform(self, tokens: List[str]) -> List[str]:
        if self.transformer:
            return self.transformer.transform(tokens)
        return tokens


@dataclass
class BuildToken2IdMixIn:

    build_dictionary: bool = False
    token2id: Token2Id = field(init=False, default=None)

    def setup_token2id(self: ITask) -> ITask:

        if self.build_dictionary:
            self.token2id = Token2Id()
            self.pipeline.payload.token2id = self.token2id

        return self

    def ingest_tokens(self, tokens: List[str]):

        if self.build_dictionary:
            self.token2id.ingest(tokens)
