from dataclasses import dataclass, field
from typing import List, Set, Union

from penelope.utility.utils import create_dataclass_instance_from_kwargs

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
    ignore_padding: bool = True
    min_window_size: int = 1
    processes: int = 6
    chunksize: int = 4
    windows_threshold: int = 1

    @property
    def props(self):
        return dict(
            context_width=self.context_width,
            concept=list(self.concept or set()),
            ignore_concept=self.ignore_concept,
            pad=self.pad,
            ignore_padding=self.ignore_padding,
            partition_keys=self.partition_keys,
            min_window_size=self.min_window_size,
            tf_threshold=self.windows_threshold,
        )

    @classmethod
    def from_kwargs(cls, **kwargs):

        instance: ContextOpts = create_dataclass_instance_from_kwargs(cls, **kwargs)

        return instance

    def get_concepts(self) -> Set[str]:
        if not self.concept:
            return set()
        return set(self.concept)
