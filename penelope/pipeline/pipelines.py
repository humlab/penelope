from typing import List

import pandas as pd
from penelope.corpus.document_index import update_document_index_statistics
from penelope.pipeline import DocumentPayload, tagged_frame_to_pos_statistics

from .pipeline import CorpusPipelineBase
from .spacy.tasks_mixin import SpacyPipelineShortcutMixIn
from .tasks_mixin import PipelineShortcutMixIn


class CorpusPipeline(
    PipelineShortcutMixIn,
    SpacyPipelineShortcutMixIn,
    CorpusPipelineBase["CorpusPipeline"],
):
    def update_statistics(self, tagged_frame: pd.DataFrame, payload: DocumentPayload, tokens: List[str]):
        """Updates document index's and payload's token statistics for given tagged_frame and result tokens."""

        # Compute statistics
        pos_statistics = tagged_frame_to_pos_statistics(tagged_frame, self.payload.pos_schema, self.get('pos_column'))

        payload.update_statistics(pos_statistics=pos_statistics, n_tokens=len(tokens))

        update_document_index_statistics(
            self.payload.document_index,
            document_name=payload.document_name,
            statistics=payload.statistics,
        )


AnyPipeline = CorpusPipeline


# class SpacyPipeline(CorpusPipeline):
#     pass
