import enum
import json
from dataclasses import dataclass
from typing import Any, Dict, Union

import pandas as pd
import yaml
from penelope.corpus.readers import TaggedTokensFilterOpts, TextReaderOpts, TextSource
from penelope.utility import get_pos_schema, replace_path

from . import interfaces


@enum.unique
class CorpusType(enum.IntEnum):
    Undefined = 0
    Text = 1
    Tokenized = 2
    SparvCSV = 3
    SpacyCSV = 4
    Pipeline = 5


@dataclass
class CorpusConfig(yaml.YAMLObject):

    # def __init__(
    #     self,
    corpus_name: str = None
    corpus_type: CorpusType = CorpusType.Undefined
    corpus_pattern: str = "*.zip"
    text_reader_opts: TextReaderOpts = None
    tagged_tokens_filter_opts: TaggedTokensFilterOpts = None
    pipeline_payload: interfaces.PipelinePayload = None
    language: str = "english"
    # ):
    #     self.corpus_name = corpus_name
    #     self.corpus_type = corpus_type
    #     self.corpus_pattern = corpus_pattern
    #     self.text_reader_opts = text_reader_opts
    #     self.tagged_tokens_filter_opts = tagged_tokens_filter_opts
    #     self.pipeline_payload = pipeline_payload
    #     self.language = language

    def folder(self, folder: str) -> "CorpusConfig":

        if isinstance(self.pipeline_payload.document_index_source, str):
            self.pipeline_payload.document_index_source = replace_path(
                self.pipeline_payload.document_index_source, folder
            )
        if isinstance(self.pipeline_payload.source, str):
            self.pipeline_payload.source = replace_path(self.pipeline_payload.source, folder)

        return self

    def files(self, source: TextSource, index_source: Union[str, pd.DataFrame]) -> "CorpusConfig":
        self.pipeline_payload.source = source
        self.pipeline_payload.document_index_source = index_source
        return self

    @property
    def pos_schema(self):
        return get_pos_schema(self.pipeline_payload.pos_schema_name)

    @property
    def props(self) -> Dict[str, Any]:
        return dict(
            corpus_name=self.corpus_name,
            corpus_type=int(self.corpus_type),
            text_reader_opts=self.text_reader_opts.props,
            pipeline_payload=self.pipeline_payload.props,
            pos_schema_name=self.pipeline_payload.pos_schema_name,
        )

    def dump(self, path: str):
        """Seserializes and writes a CorpusConfig to `path`"""
        with open(path, "w") as fp:
            if path.endswith("json"):
                json.dump(self, fp, default=vars, indent=4)
            if path.endswith('yaml') or path.endswith('yml'):
                yaml.dump(
                    json.loads(json.dumps(self, default=vars)), fp, indent=4, default_flow_style=False, sort_keys=False
                )

    @staticmethod
    def load(path: str) -> "CorpusConfig":
        """Reads and deserializes a CorpusConfig from `path`"""
        with open(path, "r") as fp:
            if path.endswith('yaml') or path.endswith('yml'):
                config_dict: dict = yaml.load(fp)
            else:
                config_dict: dict = json.load(fp)
        deserialized_config = CorpusConfig.dict_to_corpus_config(config_dict)
        return deserialized_config

    @staticmethod
    def dict_to_corpus_config(config_dict: dict) -> "CorpusConfig":

        if config_dict.get('text_reader_opts', None) is not None:
            config_dict['text_reader_opts'] = TextReaderOpts(**config_dict['text_reader_opts'])

        if config_dict.get('tagged_tokens_filter_opts', None) is not None:
            opts = config_dict['tagged_tokens_filter_opts']
            if opts.get('data', None) is not None:
                config_dict['tagged_tokens_filter_opts'] = TaggedTokensFilterOpts(**opts['data'])

        config_dict['pipeline_payload'] = interfaces.PipelinePayload(**config_dict['pipeline_payload'])

        deserialized_config: CorpusConfig = CorpusConfig(**config_dict)
        deserialized_config.corpus_type = CorpusType(deserialized_config.corpus_type)
        return deserialized_config
