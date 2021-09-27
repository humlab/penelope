import glob
import importlib
from os.path import basename, dirname, join

from .co_occurrence import pipeline_mixin as coo_mixin
from .pipeline import CorpusPipelineBase
from .pipeline_mixin import PipelineShortcutMixIn
from .spacy import pipeline_mixin as spacy_mixin
from .tasks import WildcardTask
from .topic_model import pipeline_mixin as tm_mixin


def register_pipeline_mixins():
    module_names = glob.glob(join(dirname(__file__), "*/pipeline_mixin*.py"))
    modules = [
        importlib.import_module(f"penelope.pipeline.{basename(dirname(f))}.pipeline_mixin") for f in module_names
    ]
    classes = [getattr(module, "PipelineShortcutMixIn") for module in modules]
    return classes


class CorpusPipeline(
    # *register_pipeline_mixins(),
    PipelineShortcutMixIn,
    spacy_mixin.PipelineShortcutMixIn,
    coo_mixin.PipelineShortcutMixIn,
    tm_mixin.PipelineShortcutMixIn,
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
