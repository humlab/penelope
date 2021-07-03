from __future__ import annotations

import os
from typing import TYPE_CHECKING

from penelope.utility import strip_extensions

from .. import config, pipelines, tasks

if TYPE_CHECKING:
    from ..interfaces import ITask


def checkpoint_folder_name(corpus_filename: str) -> str:
    folder, filename = os.path.split(corpus_filename)
    return os.path.join(folder, "shared", "checkpoints", f'{strip_extensions(filename)}_feather')


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

    if enable_checkpoint and not corpus_config.checkpoint_opts.feather_folder:
        checkpoint_folder: str = checkpoint_folder_name(corpus_filename)
        p = p.checkpoint_feather(checkpoint_folder, force=force_checkpoint)

    return p
