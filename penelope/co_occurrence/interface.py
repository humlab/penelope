from dataclasses import dataclass, field
from typing import Set, Union

import pandas as pd
from penelope.corpus import DocumentIndex, Token2Id
from penelope.utility import getLogger
from prometheus_client import Counter

logger = getLogger('penelope')

Token = Union[int, str]


class CoOccurrenceError(ValueError):
    ...


class PartitionKeyNotUniqueKey(ValueError):
    ...


@dataclass
class CoOccurrenceComputeResult:

    co_occurrences: pd.DataFrame = None
    document_index: DocumentIndex = None
    token2id: Token2Id = None
    token_window_counts: Counter = None


@dataclass
class ContextOpts:

    context_width: int = 2
    concept: Set[Token] = field(default_factory=set)
    ignore_concept: bool = False
    pad: Token = field(default="*")

    @property
    def props(self):
        return dict(
            context_width=self.context_width,
            concept=list(self.concept),
            ignore_concept=self.ignore_concept,
            padding=self.pad,
        )
