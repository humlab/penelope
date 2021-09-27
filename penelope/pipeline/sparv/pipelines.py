from __future__ import annotations

from typing import TYPE_CHECKING

from .. import config, pipelines, tasks

if TYPE_CHECKING:
    from ..interfaces import ITask


def to_tagged_frame_pipeline(
    *,
    corpus_config: config.CorpusConfig,
    corpus_filename: str = None,
    enable_checkpoint: bool = False,
    force_checkpoint: bool = False,
    **_,
):
    """Loads a tagged data frame"""
    p: pipelines.CorpusPipeline = pipelines.CorpusPipeline(config=corpus_config)

    corpus_filename: str = corpus_filename or corpus_config.pipeline_payload.source

    if corpus_config.corpus_type == config.CorpusType.SparvCSV:

        task: ITask = tasks.LoadTaggedCSV(
            filename=corpus_filename,
            checkpoint_opts=corpus_config.checkpoint_opts,
            extra_reader_opts=corpus_config.text_reader_opts,
        )

    elif corpus_config.corpus_type == config.CorpusType.SparvXML:
        task: ITask = tasks.LoadTaggedXML()
    else:
        raise ValueError("fatal: corpus type is not (yet) supported")

    p.add(task)

    if enable_checkpoint:
        """NOTE! If self.checkpoint_opts.feather_folder is set then LoadTaggedCSV handles feather files"""
        if not corpus_config.checkpoint_opts.feather_folder:
            p = p.checkpoint_feather(folder=corpus_config.get_feather_folder(corpus_filename), force=force_checkpoint)

    return p
