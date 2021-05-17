from .pipeline import CorpusPipelineBase
from .pipeline_mixin import PipelineShortcutMixIn
from .spacy.tasks_mixin import SpacyPipelineShortcutMixIn
from .tasks import WildcardTask


class CorpusPipeline(
    PipelineShortcutMixIn,
    SpacyPipelineShortcutMixIn,
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
