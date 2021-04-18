from __future__ import annotations

import enum
import glob
import json
import os
import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union

import yaml
from penelope.corpus.readers import TextReaderOpts
from penelope.utility import PoS_Tag_Scheme, PropertyValueMaskingOpts, create_instance, get_pos_schema

from . import checkpoint, interfaces

if TYPE_CHECKING:
    from .pipelines import CorpusPipeline


def create_pipeline_factory(
    class_or_function_name: str,
) -> Union[Callable[[CorpusConfig], CorpusPipeline], Type[CorpusPipeline]]:
    """Returns a CorpusPipeline type (class or callable that return instance) by name"""
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
class CorpusConfig:

    corpus_name: str = None
    corpus_type: CorpusType = CorpusType.Undefined
    corpus_pattern: str = "*.zip"
    checkpoint_opts: Optional[checkpoint.CheckpointOpts] = None
    text_reader_opts: TextReaderOpts = None
    tagged_tokens_filter_opts: PropertyValueMaskingOpts = None
    pipelines: dict = None
    pipeline_payload: interfaces.PipelinePayload = None
    language: str = "english"

    def get_pipeline(self, pipeline_key: str, *args, **kwargs) -> CorpusPipeline:
        """Returns a pipeline class by key from `pipelines` section"""
        if pipeline_key not in self.pipelines:
            raise ValueError(f"request of unknown pipeline failed: {pipeline_key}")
        factory: Type[CorpusPipeline] = create_pipeline_factory(self.pipelines[pipeline_key])
        return factory(self, *args, **kwargs)

    @property
    def pos_schema(self) -> PoS_Tag_Scheme:
        """Returns the part-of-speech schema"""
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
        """Return YAML filenames in given `folder`"""
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
        """Deserializes a CorpusConfig from `data_str`"""
        deserialized_config = CorpusConfig.dict_to_corpus_config(yaml.load(data_str, Loader=yaml.FullLoader))
        return deserialized_config

    @staticmethod
    def dict_to_corpus_config(config_dict: dict) -> "CorpusConfig":
        """Maps a dict read from file to a CorpusConfig instance"""
        # FIXME #47 Check if logic needs to be updated
        if config_dict.get('text_reader_opts', None) is not None:
            config_dict['text_reader_opts'] = TextReaderOpts(**config_dict['text_reader_opts'])

        if config_dict.get('tagged_tokens_filter_opts', None) is not None:
            opts = config_dict['tagged_tokens_filter_opts']
            if opts.get('data', None) is not None:
                config_dict['tagged_tokens_filter_opts'] = PropertyValueMaskingOpts(**opts['data'])

        config_dict['pipeline_payload'] = interfaces.PipelinePayload(**config_dict['pipeline_payload'])
        config_dict['checkpoint_opts'] = checkpoint.CheckpointOpts(**(config_dict.get('checkpoint_opts', {}) or {}))
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

    def folders(self, path: str, method: str = "join") -> "CorpusConfig":
        """Replaces (any) existing source path specification for corpus/index to `path`"""
        self.pipeline_payload.folders(path, method=method)
        return self
