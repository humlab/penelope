from __future__ import annotations

import contextlib
import enum
import glob
import json
import os
import pathlib
import uuid
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Type, Union

import yaml

from penelope.corpus import TextReaderOpts, TextTransformOpts
from penelope.utility import PoS_Tag_Scheme, create_instance, get_pos_schema, strip_extensions
from penelope.utility.filename_utils import replace_extension

from . import checkpoint, interfaces

if TYPE_CHECKING:
    from .pipelines import CorpusPipeline

jj = os.path.join

# pylint: disable=too-many-arguments


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

    corpus_name: str
    corpus_type: CorpusType
    corpus_pattern: str
    checkpoint_opts: Optional[checkpoint.CheckpointOpts]
    text_reader_opts: TextReaderOpts
    text_transform_opts: TextTransformOpts
    pipelines: dict
    pipeline_payload: interfaces.PipelinePayload
    language: str

    def pipeline_key_exists(self, pipeline_key: str) -> bool:
        return pipeline_key in self.pipelines

    def get_pipeline_opts(self, pipeline_key: str) -> dict:
        if pipeline_key not in self.pipelines:
            return {}
        cfg: str | dict = self.pipelines[pipeline_key]
        if isinstance(cfg, dict):
            return cfg.get('options', {})
        return {}

    def get_pipeline_opts_value(self, pipeline_key: str, key: str, default_value: Any = None) -> Any:
        return self.get_pipeline_opts(pipeline_key=pipeline_key).get(key, default_value)

    def get_pipeline_cls(self, pipeline_key: str) -> CorpusPipeline:
        """Returns a pipeline class by key from `pipelines` section"""
        if pipeline_key not in self.pipelines:
            raise ValueError(f"request of unknown pipeline failed: {pipeline_key}")
        cfg: str | dict = self.pipelines[pipeline_key]
        opts: dict = {}
        if isinstance(cfg, dict):
            if 'class_name' not in cfg:
                raise ValueError("config error: pipeline class `class_name` not specified")
            class_name: str = cfg.get('class_name')
            if 'options' in cfg:
                opts = cfg['options']
        else:
            class_name = cfg
        ctor: Type[CorpusPipeline] = create_pipeline_factory(class_name)
        return ctor, opts

    def get_pipeline(self, pipeline_key: str, **kwargs) -> CorpusPipeline:
        """Returns an instance of a pipeline class specified by key in the `pipelines` section"""
        ctor, opts = self.get_pipeline_cls(pipeline_key)
        return ctor(corpus_config=self, **opts, **kwargs)

    @property
    def pos_schema(self) -> PoS_Tag_Scheme:
        """Returns the part-of-speech schema"""
        return get_pos_schema(self.pipeline_payload.pos_schema_name)

    @property
    def props(self) -> dict[str, Any]:
        return dict(
            corpus_name=self.corpus_name,
            corpus_type=int(self.corpus_type),
            corpus_pattern=self.corpus_pattern,
            checkpoint_opts=asdict(self.checkpoint_opts) if self.checkpoint_opts else None,
            text_reader_opts=asdict(self.text_reader_opts) if self.text_reader_opts else None,
            text_transform_opts=self.text_transform_opts.props if self.text_transform_opts else None,
            pipeline_payload=self.pipeline_payload.props,
            pipelines=self.pipelines,
            # pos_schema_name=self.pipeline_payload.pos_schema_name,
            language=self.language,
        )

    def dump(self, path: str):
        """Serializes and writes a CorpusConfig to `path`"""
        # memory_store: dict = self.pipeline_payload.memory_store
        # self.pipeline_payload.memory_store = None

        with open(path, "w") as fp:
            if path.endswith("json"):
                json.dump(self, fp, default=vars, indent=4, allow_nan=True)
            if path.endswith('yaml') or path.endswith('yml'):
                yaml.dump(
                    # json.loads(json.dumps(self, default=vars)),
                    self.props,
                    fp,
                    indent=4,
                    default_flow_style=False,
                    sort_keys=False,
                )
        # self.pipeline_payload.memory_store = memory_store

    @staticmethod
    def list(folder: str) -> list[str]:
        """Return YAML filenames in given `folder`"""
        filenames = sorted(
            glob.glob(jj(folder, '**', '*.yml'), recursive=True) + glob.glob(jj(folder, '**', '*.yaml'), recursive=True)
        )
        return filenames

    @staticmethod
    def load(path: str, source: Optional[str] = None) -> "CorpusConfig":
        """Reads and deserializes a CorpusConfig from `path`"""
        with open(path, "r") as fp:
            if path.endswith('yaml') or path.endswith('yml'):
                config_dict: dict = yaml.load(fp, Loader=yaml.FullLoader)
            else:
                config_dict: dict = json.load(fp)
        deserialized_config = CorpusConfig.dict_to_corpus_config(config_dict)
        if source is not None:
            deserialized_config.pipeline_payload.source = source
        return deserialized_config

    @staticmethod
    def loads(data_str: str) -> "CorpusConfig":
        """Deserializes a CorpusConfig from `data_str`"""
        deserialized_config = CorpusConfig.dict_to_corpus_config(yaml.load(data_str, Loader=yaml.FullLoader))
        return deserialized_config

    @staticmethod
    def dict_to_corpus_config(config_dict: dict) -> "CorpusConfig":
        """Maps a dict read from file to a CorpusConfig instance"""

        """Remove deprecated key"""
        if 'filter_opts' in config_dict:
            del config_dict['filter_opts']

        if config_dict.get('text_reader_opts', None) is not None:
            config_dict['text_reader_opts'] = TextReaderOpts(**config_dict['text_reader_opts'])

        if config_dict.get('text_transform_opts', None) is not None:
            config_dict['text_transform_opts'] = TextTransformOpts(**config_dict['text_transform_opts'])

        config_dict['pipeline_payload'] = interfaces.PipelinePayload(**config_dict['pipeline_payload'])
        config_dict['checkpoint_opts'] = checkpoint.CheckpointOpts(**(config_dict.get('checkpoint_opts', {}) or {}))
        config_dict['pipelines'] = config_dict.get(
            'pipelines', {}
        )  # CorpusConfig.dict_to_pipeline_config(config_dict.get('pipelines', {}))

        deserialized_config: CorpusConfig = CorpusConfig.create(**config_dict)
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
            try_name: str = filename if not extension else replace_extension(filename, extension=extension)
            candidates: list[pathlib.Path] = list(pathlib.Path(folder).rglob(try_name))
            try:
                for candidate in candidates:
                    config = CorpusConfig.load(str(candidate))
                    return config
            except:  # pylint: disable=bare-except
                pass
        raise FileNotFoundError(filename)

    @staticmethod
    def find_all(folder: str) -> list["CorpusConfig"]:
        """Finds all corpus configs in `folder`"""

        configs: list[CorpusConfig] = []

        if not os.path.isdir(folder):
            return configs

        candidates: list[str] = glob.glob(jj(folder, "*.yml")) + glob.glob(jj(folder, "*.yaml"))

        for candidate in candidates:
            with contextlib.suppress(Exception):
                config = CorpusConfig.load(candidate)
                configs.append(config)

        return configs

    def folders(self, path: str, method: Literal['join', 'replace'] = "join") -> "CorpusConfig":
        """Replaces (any) existing source path specification for corpus/index to `path`"""
        self.pipeline_payload.folders(path, method=method)
        return self

    @staticmethod
    def tokenized_corpus_config(language: str = "swedish") -> CorpusConfig:
        config: CorpusConfig = CorpusConfig.create(
            corpus_name=uuid.uuid1(),
            corpus_type=CorpusType.Tokenized,
            corpus_pattern=None,
            checkpoint_opts=None,
            text_reader_opts=None,
            text_transform_opts=None,
            pipelines=None,
            pipeline_payload=interfaces.PipelinePayload(),
            language=language,
        )
        return config

    def get_feather_folder(self, corpus_source: str | None) -> str | None:

        if self.checkpoint_opts.feather_folder is not None:
            return self.checkpoint_opts.feather_folder

        corpus_source: str = corpus_source or self.pipeline_payload.source

        if corpus_source is None:
            return None

        folder, filename = os.path.split(corpus_source)
        return jj(folder, "shared", "checkpoints", f'{strip_extensions(filename)}_feather')

    def corpus_source_exists(self):
        if self.pipeline_payload.source is None:
            return False
        return os.path.isfile(self.pipeline_payload.source)

    @staticmethod
    def create(
        corpus_name: str = None,
        corpus_type: CorpusType = CorpusType.Undefined,
        corpus_pattern: str = "*.zip",
        checkpoint_opts: Optional[checkpoint.CheckpointOpts] = None,
        text_reader_opts: TextReaderOpts = None,
        text_transform_opts: TextTransformOpts = None,
        pipelines: dict = None,
        pipeline_payload: interfaces.PipelinePayload = None,
        language: str = "english",
    ) -> CorpusConfig:

        return CorpusConfig(
            corpus_name=corpus_name,
            corpus_type=corpus_type,
            corpus_pattern=corpus_pattern,
            checkpoint_opts=checkpoint_opts,
            text_reader_opts=text_reader_opts,
            text_transform_opts=text_transform_opts,
            pipelines=pipelines,
            pipeline_payload=pipeline_payload,
            language=language,
        )
