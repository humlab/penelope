import contextlib
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from penelope.corpus import Token2Id, TokensTransformer, TokensTransformOpts
from penelope.corpus.document_index import update_document_index_by_dicts_or_tuples
from penelope.utility import PoS_Tag_Scheme
from penelope.utility.file_utility import write_json

from . import interfaces

# pylint: disable=no-member


class DefaultResolveMixIn:
    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        return payload


class PoSCountError(ValueError):
    ...


@dataclass
class TokenCountMixIn:

    token_counts: dict = field(init=False, default_factory=list)

    def register_token_count(self, document_name: str, n_tokens: int) -> None:
        self.token_counts.append((document_name, n_tokens))

    def flush_token_counts(self, document_index: pd.DataFrame) -> None:
        update_document_index_by_dicts_or_tuples(
            document_index, data=self.token_counts, columns=['n_tokens'], dtype=np.int32, default=0
        )

    def exit_hook(self):
        self.flush_token_counts(self.document_index)


@dataclass
class PoSCountMixIn:

    pos_column: str = None
    document_tfs: Mapping[str, Tuple[Any, ...]] = field(init=False, default_factory=dict)
    enable_pos_counts: bool = True

    # def __post_init__(self):
    #     self.exit_hooks.append(self.flush_pos_counts2)

    def register_pos_counts2(
        self, *, document_name: str, tagged_frame: pd.DataFrame, pos_schema: PoS_Tag_Scheme
    ) -> None:
        """Computes count per PoS group. Store in local property."""
        if not self.enable_pos_counts:
            return

        if not isinstance(tagged_frame, pd.DataFrame):
            raise PoSCountError(f"{document_name}: config error (not a tagged frame)")

        if self.pos_column is None:
            self.pos_column = 'pos_id' if 'pos_id' in tagged_frame.columns else 'pos'

        pos_counts: dict = pos_schema.PoS_group_counts(tagged_frame[self.pos_column])
        pos_counts.update(document_name=document_name)
        self.document_tfs[document_name] = pos_counts

    def register_pos_counts(self, payload: interfaces.DocumentPayload) -> None:
        """Computes token counts from the tagged frame, and adds them to the document index"""
        if not self.enable_pos_counts:
            return

        self.register_pos_counts2(
            document_name=payload.document_name,
            tagged_frame=payload.content,
            pos_schema=self.pipeline.payload.pos_schema,
        )

    def flush_pos_counts(self) -> None:
        """Flushes token counts to document index (inplace)."""
        if not self.enable_pos_counts:
            return

        self.flush_pos_counts2(document_index=self.document_index)

    def flush_pos_counts2(self, document_index: pd.DataFrame) -> None:
        """Flushes token counts to document index (inplace)."""
        if not self.enable_pos_counts:
            return

        if len(self.document_tfs) > 0:
            update_document_index_by_dicts_or_tuples(document_index=document_index, data=self.document_tfs)

    def store_pos_counts(self, filename: str = 'document_tfs.json') -> None:
        if not self.enable_pos_counts:
            return

        with contextlib.suppress(Exception):
            write_json(filename, self.document_tfs)

    def exit_hook(self):
        self.flush_pos_counts()


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
