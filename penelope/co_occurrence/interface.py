from dataclasses import dataclass, field
from typing import List, Set, Union

from penelope.utility import getLogger

logger = getLogger('penelope')

Token = Union[int, str]


class ZeroComputeError(ValueError):
    def __init__(self):
        super().__init__("Computation ended up in ZERO records. Check settings!")


class CoOccurrenceError(ValueError):
    ...


class PartitionKeyNotUniqueKey(ValueError):
    ...


@dataclass
class ContextOpts:

    context_width: int = 2
    concept: Set[Token] = field(default_factory=set)
    ignore_concept: bool = False
    pad: Token = field(default="*")
    partition_keys: List[str] = field(default_factory=list)
    ignore_padding: bool = False

    @property
    def props(self):
        return dict(
            context_width=self.context_width,
            concept=list(self.concept),
            ignore_concept=self.ignore_concept,
            padding=self.pad,
            ignore_padding=self.ignore_padding,
            partition_keys=self.partition_keys,
        )
