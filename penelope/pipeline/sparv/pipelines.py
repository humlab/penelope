from .. import config, interfaces, pipelines, tasks


def to_tagged_frame_pipeline(corpus_config: config.CorpusConfig, **_):
    """Loads a teagged data frame"""
    p: pipelines.CorpusPipeline = pipelines.CorpusPipeline(config=corpus_config)

    if corpus_config.corpus_type == config.CorpusType.SparvCSV:
        task: interfaces.ITask = tasks.LoadTaggedCSV(
            filename=corpus_config.pipeline_payload.source,
            options=corpus_config.content_deserialize_opts,
            extra_reader_opts=corpus_config.text_reader_opts,
        )

    elif corpus_config.corpus_type == config.CorpusType.SparvXML:
        task: interfaces.ITask = tasks.LoadTaggedXML()
    else:
        raise ValueError("fatal: corpus type is not (yet) supported")

    p.add(task)
    return p
