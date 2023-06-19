import glob
import importlib
from os.path import basename, dirname, join

from penelope import corpus as pc
from penelope import utility as pu
from penelope.pipeline.interfaces import DEFAULT_TAGGED_FRAMES_FILENAME_SUFFIX

from .co_occurrence import pipeline_mixin as coo_mixin
from .config import CorpusConfig
from .dtm import pipeline_mixin as dtm_mixin
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
    PipelineShortcutMixIn,
    spacy_mixin.PipelineShortcutMixIn,
    coo_mixin.PipelineShortcutMixIn,
    tm_mixin.PipelineShortcutMixIn,
    dtm_mixin.PipelineShortcutMixIn,
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


def to_tagged_frame_pipeline(
    *,
    corpus_config: CorpusConfig,
    corpus_source: str = None,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
    tagged_corpus_source: str = None,
    text_transform_opts: pc.TextTransformOpts = None,
    **_,
) -> CorpusPipeline:
    """Tag corpus using spaCy pipeline. Store result as tagged (pos) data frames

    Args:
        corpus_config (CorpusConfig): Corpus config
        corpus_source (str, optional): Corpus source. Defaults to None.
        enable_checkpoint (bool, optional): Store checkpoint of tagged corpus. Defaults to True.
        force_checkpoint (bool, optional): Force checkpoint if exists. Defaults to False.
        tagged_corpus_source (str, optional): Tagged corpus target name. Defaults to None.
        text_transform_opts (TextTransformOpts, optional): Text transforms to apply ahead of tagging. Defaults to None.

    Raises:
        ex: [description]

    Returns:
        CorpusPipeline: [description]
    """
    try:
        tagged_frame_filename: str = tagged_corpus_source or pu.path_add_suffix(
            corpus_config.pipeline_payload.source, DEFAULT_TAGGED_FRAMES_FILENAME_SUFFIX
        )

        pipeline: CorpusPipeline = (
            CorpusPipeline(config=corpus_config)
            .load_text(
                reader_opts=corpus_config.text_reader_opts,
                transform_opts=text_transform_opts or corpus_config.text_transform_opts,
                source=corpus_source,
            )
            .to_tagged_frame(tagger=corpus_config.resolve_dependency("tagger"))
            .checkpoint(filename=tagged_frame_filename, force_checkpoint=force_checkpoint)
        )

        if enable_checkpoint:
            pipeline = pipeline.checkpoint_feather(
                folder=corpus_config.get_feather_folder(corpus_source), force=force_checkpoint
            )

        return pipeline

    except Exception as ex:
        raise ex
