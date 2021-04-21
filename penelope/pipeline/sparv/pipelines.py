import os

from penelope.utility import strip_extensions

from .. import config, interfaces, pipelines, tasks


def checkpoint_folder_name(corpus_filename: str) -> str:
    folder, filename = os.path.split(corpus_filename)
    return os.path.join(folder, "shared", "checkpoints", f'{strip_extensions(filename)}_feather')


def to_tagged_frame_pipeline(
    corpus_config: config.CorpusConfig,
    corpus_filename: str = None,
    force_checkpoint: str = None,
    **_,
):
    """Loads a tagged data frame"""
    p: pipelines.CorpusPipeline = pipelines.CorpusPipeline(config=corpus_config)

    corpus_filename: str = corpus_filename or corpus_config.pipeline_payload.source
    checkpoint_folder: str = checkpoint_folder_name(corpus_filename)

    if corpus_config.corpus_type == config.CorpusType.SparvCSV:

        task: interfaces.ITask = tasks.LoadTaggedCSV(
            filename=corpus_filename,
            checkpoint_opts=corpus_config.checkpoint_opts,
            extra_reader_opts=corpus_config.text_reader_opts,
        )

    elif corpus_config.corpus_type == config.CorpusType.SparvXML:
        task: interfaces.ITask = tasks.LoadTaggedXML()
    else:
        raise ValueError("fatal: corpus type is not (yet) supported")

    p.add(task).checkpoint_feather(checkpoint_folder, force=force_checkpoint)

    return p
