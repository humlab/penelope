from __future__ import annotations

from typing import TYPE_CHECKING

from .. import config, pipelines, tasks

if TYPE_CHECKING:
    from ..interfaces import ITask


def to_tagged_frame_pipeline(
    *,
    corpus_config: config.CorpusConfig,
    corpus_source: str = None,
    enable_checkpoint: bool = False,
    force_checkpoint: bool = False,
    enable_pos_counts: bool = True,
    stop_at_index: int = None,
    **_,
):
    """Loads a tagged data frame"""
    p: pipelines.CorpusPipeline = pipelines.CorpusPipeline(config=corpus_config)

    corpus_source: str = corpus_source or corpus_config.pipeline_payload.source

    if corpus_config.corpus_type == config.CorpusType.SparvCSV:

        task: ITask = tasks.LoadTaggedCSV(
            filename=corpus_source,
            checkpoint_opts=corpus_config.checkpoint_opts,
            extra_reader_opts=corpus_config.text_reader_opts,
            stop_at_index=stop_at_index,
            enable_counts=enable_pos_counts,
        )

    elif corpus_config.corpus_type == config.CorpusType.SparvXML:
        task: ITask = tasks.LoadTaggedXML()
    else:
        raise ValueError("fatal: corpus type is not (yet) supported")

    p.add(task)

    if enable_checkpoint:
        """NOTE! If self.checkpoint_opts.feather_folder is set then LoadTaggedCSV handles feather files"""
        if not corpus_config.checkpoint_opts.feather_folder:
            p = p.checkpoint_feather(folder=corpus_config.get_feather_folder(corpus_source), force=force_checkpoint)

    return p
