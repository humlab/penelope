from dataclasses import dataclass, field
from typing import Set

CO_OCCURRENCE_FILENAME_POSTFIX = '_co-occurrence.csv.zip'


class CoOccurrenceError(ValueError):
    pass


@dataclass
class ContextOpts:

    context_width: int = 2
    concept: Set[str] = field(default_factory=set)
    ignore_concept: bool = False

    @property
    def props(self):
        return dict(context_width=self.context_width, concept=list(self.concept), ignore_concept=self.ignore_concept)
