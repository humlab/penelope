import logging

from . import convert
from .interfaces import DocumentPayload


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
