import logging

import pandas as pd

from . import convert
from .interfaces import DocumentPayload


class DefaultResolveMixIn:
    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        return payload


class UpdateDocumentPropertyMixIn:
    def store_token_counts(self, payload: DocumentPayload, tagged_frame: pd.DataFrame):
        """Computes token counts from the tagged frame, and adds them to the document index"""
        try:
            pos_column = self.pipeline.payload.get('pos_column')
            token_counts = convert.tagged_frame_to_token_counts(
                tagged_frame, self.pipeline.payload.pos_schema, pos_column
            )
            self.store_document_properties(payload, **token_counts)
        except Exception as ex:
            logging.exception(ex)
            raise
