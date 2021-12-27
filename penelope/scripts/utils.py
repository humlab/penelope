from os.path import dirname, isdir, isfile
from typing import Optional

from penelope import pipeline
from penelope.utility import update_dict_from_yaml_file


def update_arguments_from_options_file(*, arguments: dict, filename_key: str) -> dict:
    options_filename: Optional[str] = arguments.get(filename_key)
    del arguments[filename_key]
    arguments = update_dict_from_yaml_file(options_filename, arguments)
    return arguments


def load_config(config_filename: str, corpus_source: str) -> pipeline.CorpusConfig:
    config: pipeline.CorpusConfig = pipeline.CorpusConfig.load(path=config_filename)
    if config.pipeline_payload.source is None:
        config.pipeline_payload.source = corpus_source
        if isdir(corpus_source):
            config.folders(corpus_source, method='replace')
        elif isfile(corpus_source):
            config.folders(dirname(corpus_source), method='replace')
    return config


def remove_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}
