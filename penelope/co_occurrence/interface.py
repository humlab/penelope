from dataclasses import dataclass, field
from typing import Set

from penelope.utility import getLogger

logger = getLogger('penelope')


class CoOccurrenceError(ValueError):
    ...


@dataclass
class ContextOpts:

    context_width: int = 2
    concept: Set[str] = field(default_factory=set)
    ignore_concept: bool = False

    @property
    def props(self):
        return dict(context_width=self.context_width, concept=list(self.concept), ignore_concept=self.ignore_concept)
