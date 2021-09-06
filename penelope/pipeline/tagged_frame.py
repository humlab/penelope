from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import pandas as pd
from penelope.utility import PoS_Tag_Scheme

from .interfaces import ITask
from .pipeline import ContentType, DocumentPayload
from .tasks import Vocabulary


class IngestVocabType(IntEnum):
    """Use supplied vocab"""

    Supplied = 0
    """Build vocab in separate pass on enter"""
    Prebuild = 1
    """Ingest tokens from each document incrementally"""
    Incremental = 2


@dataclass
class ToIdTaggedFrame(Vocabulary):
    """Encode a TaggedFrame to a numerical representation.
    Return data frame with columns `token_id` and `pos_id`.
    """

    ingest_vocab_type: str = field(default=IngestVocabType.Incremental)

    def __post_init__(self):

        super().__post_init__()

        self.in_content_type = ContentType.TAGGED_FRAME
        self.out_content_type = ContentType.TAGGED_ID_FRAME

    def setup(self) -> ITask:

        if self.ingest_vocab_type == IngestVocabType.Supplied:

            self.token2id = self.token2id or self.pipeline.payload.token2id

            if self.token2id is None or len(self.token2id) == 0:
                raise ValueError("Non-empty token2id must be supplied when ingest type is Supplied")

            if self.pipeline.payload.token2id is not self.token2id:
                self.pipeline.payload.token2id = self.token2id

        return super().setup()

    def enter(self):

        if self.ingest_vocab_type == IngestVocabType.Prebuild:
            super().enter()

        if self.ingest_vocab_type == IngestVocabType.Incremental:
            self.token2id.ingest(self.token2id.magic_tokens)

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:

        tagged_frame: pd.DataFrame = payload.content

        pos_schema: PoS_Tag_Scheme = self.pipeline.config.pipeline_payload.pos_schema
        token_column: str = self.target
        pos_column: str = self.pipeline.get('pos_column', None)

        if self.ingest_vocab_type == IngestVocabType.Incremental:
            self.token2id.ingest(tagged_frame[token_column])  # type: ignore

        id_tagged_frame: pd.DataFrame = pd.DataFrame(
            data=dict(
                token_id=tagged_frame[token_column].map(self.token2id).astype(np.int32),
                pos_id=tagged_frame[pos_column].map(pos_schema.pos_to_id).astype(np.int8),
            )
        )
        return payload.update(ContentType.TAGGED_ID_FRAME, id_tagged_frame)
