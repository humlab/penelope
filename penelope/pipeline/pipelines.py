from . import pipeline, tasks_mixin
from .spacy import tasks_mixin as spacy_tasks_mixin


class CorpusPipeline(
    tasks_mixin.PipelineShortcutMixIn,
    spacy_tasks_mixin.SpacyPipelineShortcutMixIn,
    pipeline.CorpusPipelineBase["CorpusPipeline"],
):
    pass


AnyPipeline = CorpusPipeline


class SpacyPipeline(CorpusPipeline):
    pass
