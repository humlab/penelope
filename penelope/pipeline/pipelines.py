from .pipeline import CorpusPipelineBase
from .spacy.tasks_mixin import SpacyPipelineShortcutMixIn
from .tasks_mixin import PipelineShortcutMixIn


class CorpusPipeline(
    PipelineShortcutMixIn,
    SpacyPipelineShortcutMixIn,
    CorpusPipelineBase["CorpusPipeline"],
):
    pass


AnyPipeline = CorpusPipeline


class SpacyPipeline(CorpusPipeline):
    pass
