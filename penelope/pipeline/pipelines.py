from .co_occurrence import pipeline_mixin as co_occurrence
from .pipeline import CorpusPipelineBase
from .pipeline_mixin import PipelineShortcutMixIn
from .spacy import pipeline_mixin as spacy
from .tasks import WildcardTask


class CorpusPipeline(
    PipelineShortcutMixIn,
    spacy.SpacyPipelineShortcutMixIn,
    co_occurrence.PipelineShortcutMixIn,
    CorpusPipelineBase["CorpusPipeline"],
):
    def __add__(self, other: "CorpusPipeline") -> "CorpusPipeline":
        if len(other.tasks) > 0:
            has_wildcard: bool = isinstance(other.tasks[0], WildcardTask)
            self.add(other.tasks[1 if has_wildcard else 0 :])
            if self.payload is not None:
                self.payload.extend(other.payload)
        return self


AnyPipeline = CorpusPipeline


def wildcard() -> CorpusPipeline:
    p: CorpusPipeline = CorpusPipeline(config=None)
    return p
