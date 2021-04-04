from __future__ import annotations

import csv
import enum
import glob
import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union

import yaml
from penelope.corpus.readers import TaggedTokensFilterOpts, TextReaderOpts
from penelope.utility import create_instance, get_pos_schema

from . import interfaces

if TYPE_CHECKING:
    from .pipelines import CorpusPipeline


def create_pipeline_factory(
    class_or_function_name: str,
) -> Union[Callable[[CorpusConfig], CorpusPipeline], Type[CorpusPipeline]]:
    factory = create_instance(class_or_function_name)
    return factory


@enum.unique
class CorpusType(enum.IntEnum):
    Undefined = 0
    Text = 1
    Tokenized = 2
    SparvCSV = 3
    SpacyCSV = 4
    Pipeline = 5
    SparvXML = 6


@dataclass
class CheckpointOpts:

    content_type_code: int = 0

    document_index_name: str = field(default="document_index.csv")
    document_index_sep: str = field(default='\t')

    sep: str = '\t'
    quoting: int = csv.QUOTE_NONE
    custom_serializer_classname: str = None

    text_column: str = field(default="text")
    lemma_column: str = field(default="lemma")
    pos_column: str = field(default="pos")
    extra_columns: List[str] = field(default_factory=list)

    @property
    def content_type(self) -> interfaces.ContentType:
        return interfaces.ContentType(self.content_type_code)

    @content_type.setter
    def content_type(self, value: interfaces.ContentType):
        self.content_type_code = int(value)

    def as_type(self, value: interfaces.ContentType) -> "CheckpointOpts":
        opts = CheckpointOpts(
            content_type_code=int(value),
            document_index_name=self.document_index_name,
            document_index_sep=self.document_index_sep,
            sep=self.sep,
            quoting=self.quoting,
        )
        return opts

    @staticmethod
    def load(data: dict) -> "CheckpointOpts":
        opts = CheckpointOpts()
        for key in data.keys():
            if hasattr(opts, key):
                setattr(opts, key, data[key])
        return opts

    @property
    def custom_serializer(self) -> type:
        if not self.custom_serializer_classname:
            return None
        return create_instance(self.custom_serializer_classname)


@dataclass
class CorpusConfig:

    corpus_name: str = None
    corpus_type: CorpusType = CorpusType.Undefined
    corpus_pattern: str = "*.zip"
    # Used when corpus data needs to be deserialized (e.g. zipped csv data etc)
    checkpoint_opts: Optional[CheckpointOpts] = None
    text_reader_opts: TextReaderOpts = None
    tagged_tokens_filter_opts: TaggedTokensFilterOpts = None
    pipelines: dict = None
    pipeline_payload: interfaces.PipelinePayload = None
    language: str = "english"

    def get_pipeline(self, pipeline_key: str, *args, **kwargs) -> Union[Callable, Type]:
        if pipeline_key not in self.pipelines:
            raise ValueError(f"request of unknown pipeline failed: {pipeline_key}")
        factory = create_pipeline_factory(self.pipelines[pipeline_key])
        return factory(self, *args, **kwargs)

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
        """Serializes and writes a CorpusConfig to `path`"""
        with open(path, "w") as fp:
            if path.endswith("json"):
                json.dump(self, fp, default=vars, indent=4)
            if path.endswith('yaml') or path.endswith('yml'):
                yaml.dump(
                    json.loads(json.dumps(self, default=vars)), fp, indent=4, default_flow_style=False, sort_keys=False
                )

    @staticmethod
    def list(folder: str) -> List[str]:
        """Return YAML filenames in `folder`"""
        filenames = sorted(glob.glob(os.path.join(folder, '*.yml')) + glob.glob(os.path.join(folder, '*.yaml')))
        return filenames

    @staticmethod
    def load(path: str) -> "CorpusConfig":
        """Reads and deserializes a CorpusConfig from `path`"""
        with open(path, "r") as fp:
            if path.endswith('yaml') or path.endswith('yml'):
                config_dict: dict = yaml.load(fp, Loader=yaml.FullLoader)
            else:
                config_dict: dict = json.load(fp)
        deserialized_config = CorpusConfig.dict_to_corpus_config(config_dict)
        return deserialized_config

    @staticmethod
    def loads(data_str: str) -> "CorpusConfig":
        """Reads and deserializes a CorpusConfig from `path`"""
        deserialized_config = CorpusConfig.dict_to_corpus_config(yaml.load(data_str, Loader=yaml.FullLoader))
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
        config_dict['checkpoint_opts'] = CheckpointOpts(
            **(config_dict.get('checkpoint_opts', {}) or {})
        )
        config_dict['pipelines'] = config_dict.get(
            'pipelines', {}
        )  # CorpusConfig.dict_to_pipeline_config(config_dict.get('pipelines', {}))

        deserialized_config: CorpusConfig = CorpusConfig(**config_dict)
        deserialized_config.corpus_type = CorpusType(deserialized_config.corpus_type)

        return deserialized_config

    @staticmethod
    def find(filename: str, folder: str) -> "CorpusConfig":
        """Finds and returns a corpus config named `filename` in `folder`"""
        if isinstance(filename, CorpusConfig):
            return filename

        if not os.path.isdir(folder):
            raise FileNotFoundError(folder)

        for extension in ['', '.yml', '.yaml', '.json']:
            try_name: str = f"{filename}{extension}"
            candidates: List[pathlib.Path] = list(pathlib.Path(folder).rglob(try_name))
            try:
                for candidate in candidates:
                    config = CorpusConfig.load(str(candidate))
                    return config
            except:  # pylint: disable=bare-except
                pass
        FileNotFoundError(filename)

    @property
    def serialize_opts(self) -> CheckpointOpts:
        opts = CheckpointOpts(
            document_index_name=self.pipeline_payload.document_index_source,
            document_index_sep=self.pipeline_payload.document_index_sep,
        )
        return opts

    def folders(self, path: str, method: str = "join") -> "CorpusConfig":
        """Replaces (any) existing source path specification for corpus/index to `path`"""
        self.pipeline_payload.folders(path, method=method)
        return self
