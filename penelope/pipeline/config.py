from __future__ import annotations

import contextlib
import enum
import glob
import json
import os
import pathlib
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Type, Union

import yaml

from penelope import utility as pu
from penelope.corpus import TextReaderOpts, TextTransformOpts
from penelope.corpus.serialize import SerializeOpts

from . import interfaces

if TYPE_CHECKING:
    from .pipelines import CorpusPipeline

jj = os.path.join

# pylint: disable=too-many-arguments, too-many-public-methods


def create_pipeline_factory(
    class_or_function_name: str,
) -> Union[Callable[[CorpusConfig], CorpusPipeline], Type[CorpusPipeline]]:
    """Returns a CorpusPipeline type (class or callable that return instance) by name"""
    factory = pu.create_class(class_or_function_name)
    return factory


class DependencyError(Exception):
    ...


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
    serialize_opts: Optional[SerializeOpts]
    text_reader_opts: TextReaderOpts
    text_transform_opts: TextTransformOpts
    pipelines: dict
    pipeline_payload: interfaces.PipelinePayload
    language: str
    extra_opts: dict[str, Any] = field(default_factory=dict)
    dependencies: dict[str, Any] = field(default_factory=dict)

    _key_container: dict[str, Any] = field(default_factory=dict, init=False)

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
    def pos_schema(self) -> pu.PoS_Tag_Scheme:
        """Returns the part-of-speech schema"""
        return pu.get_pos_schema(self.pipeline_payload.pos_schema_name)

    @property
    def props(self) -> dict[str, Any]:
        return dict(
            corpus_name=self.corpus_name,
            corpus_type=int(self.corpus_type),
            corpus_pattern=self.corpus_pattern,
            serialize_opts=asdict(self.serialize_opts) if self.serialize_opts else None,
            text_reader_opts=asdict(self.text_reader_opts) if self.text_reader_opts else None,
            text_transform_opts=str(self.text_transform_opts.transforms) if self.text_transform_opts else None,
            pipeline_payload=self.pipeline_payload.props,
            pipelines=self.pipelines,
            language=self.language,
            extra_opts=self.extra_opts,
            dependencies=self.dependencies,
        )

    def dump(self, path: str) -> None:
        """Serializes and writes a CorpusConfig to `path`"""

        with open(path, "w", encoding="utf-8") as fp:
            if path.endswith("json"):
                json.dump(self, fp, default=vars, indent=4, allow_nan=True)
            if path.endswith('yaml') or path.endswith('yml'):
                yaml.dump(self.props, fp, indent=2, default_flow_style=False, sort_keys=False, encoding='utf-8')

    @staticmethod
    def load(path: str, source: Optional[str] = None) -> "CorpusConfig":
        """Reads and deserializes a CorpusConfig from `path`"""
        with open(path, "r") as fp:
            if path.endswith('yaml') or path.endswith('yml'):
                config_dict: dict = yaml.load(fp, Loader=yaml.FullLoader)
            else:
                config_dict: dict = json.load(fp)
        deserialized_config: CorpusConfig = CorpusConfig.dict_to_corpus_config(config_dict)
        if source is not None:
            deserialized_config.pipeline_payload.source = source
        return deserialized_config

    @staticmethod
    def loads(data_str: str) -> "CorpusConfig":
        """Deserializes a CorpusConfig from `data_str`"""
        deserialized_config: CorpusConfig = CorpusConfig.dict_to_corpus_config(
            yaml.load(data_str, Loader=yaml.FullLoader)
        )
        return deserialized_config

    @staticmethod
    def decode_transform_opts(transform_opts: dict) -> TextTransformOpts:
        if transform_opts is None:
            return None
        if isinstance(transform_opts, (str, pu.CommaStr, dict)):
            return TextTransformOpts(transforms=transform_opts)
        return None

    @staticmethod
    def dict_to_corpus_config(config_dict: dict) -> "CorpusConfig":
        """Maps a dict read from file to a CorpusConfig instance"""

        if 'corpus_name' not in config_dict:
            raise ValueError("CorpusConfig load failed. Mandatory key 'corpus_name' is missing.")

        if config_dict.get('text_reader_opts', None) is not None:
            config_dict['text_reader_opts'] = TextReaderOpts(**config_dict['text_reader_opts'])

        transform_opts: dict | str = config_dict.get('text_transform_opts', None)
        if transform_opts is not None:
            config_dict['text_transform_opts'] = CorpusConfig.decode_transform_opts(transform_opts)

        config_dict['pipeline_payload'] = interfaces.PipelinePayload(**config_dict['pipeline_payload'])
        config_dict['serialize_opts'] = SerializeOpts(
            **(config_dict.get('serialize_opts', {}) or config_dict.get('checkpoint_opts', {}) or {})
        )
        config_dict['pipelines'] = config_dict.get('pipelines', {})
        config_dict['dependencies'] = config_dict.get('dependencies', {})
        config_dict['extra_opts'] = config_dict.get('extra_opts', {})

        keys_to_keep: list[str] = pu.get_func_args(CorpusConfig.create)
        config_dict = {k: v for k, v in config_dict.items() if k in keys_to_keep}

        deserialized_config: CorpusConfig = CorpusConfig.create(**config_dict)
        deserialized_config.corpus_type = CorpusType(deserialized_config.corpus_type)

        return deserialized_config

    @staticmethod
    def find(filename: str, folder: str, set_folder: bool = False) -> "CorpusConfig":
        """Finds and returns a corpus config named `filename` in `folder`"""
        if isinstance(filename, CorpusConfig):
            return filename

        if not os.path.isdir(folder):
            raise FileNotFoundError(folder)

        for extension in ['', '.yml', '.yaml', '.json']:
            try_name: str = filename if not extension else pu.replace_extension(filename, extension=extension)
            candidates: list[pathlib.Path] = list(pathlib.Path(folder).rglob(try_name))
            try:
                for candidate in candidates:
                    config: CorpusConfig = CorpusConfig.load(str(candidate))
                    if set_folder:
                        config.pipeline_payload.folders(folder)
                    return config
            except:  # pylint: disable=bare-except
                pass
        raise FileNotFoundError(filename)

    @staticmethod
    def find_all(folder: str, recursive: bool = False, set_folder: bool = False) -> list["CorpusConfig"]:
        """Finds all corpus configs in `folder`"""

        configs: list[CorpusConfig] = []

        if not os.path.isdir(folder):
            return configs

        for pattern in ["*.yml", "*.yaml"]:
            candidates: list[str] = glob.glob(jj(folder, "**" if recursive else "", pattern), recursive=recursive)
            for candidate in candidates:
                with contextlib.suppress(Exception):
                    config = CorpusConfig.load(candidate)
                    if set_folder:
                        """Update pipeline_payload.source to match folder"""
                        config.pipeline_payload.folders(os.path.dirname(candidate))
                    configs.append(config)

        return configs

    @staticmethod
    def list_all(folder: str, recursive: bool = False, try_load: bool = False) -> list[str]:
        """Return YAML filenames in given `folder`"""
        filenames: list[str] = [
            c
            for pattern in ["*.yml", "*.yaml"]
            for c in glob.glob(jj(folder, "**" if recursive else "", pattern), recursive=recursive)
        ]
        if try_load:
            return [filename for filename in filenames if CorpusConfig.is_config(filename)]
        return filenames

    @staticmethod
    def is_config(path: str) -> bool:
        """Returns True if `path` can be loaded as a CorpusConfig"""
        with contextlib.suppress(Exception):
            CorpusConfig.load(path)
            return True
        return False

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
            serialize_opts=None,
            text_reader_opts=None,
            text_transform_opts=None,
            pipelines=None,
            pipeline_payload=interfaces.PipelinePayload(),
            language=language,
            dependencies={},
            extra_opts={},
        )
        return config

    def get_feather_folder(self, corpus_source: str | None) -> str | None:
        if self.serialize_opts.feather_folder is not None:
            return self.serialize_opts.feather_folder

        corpus_source: str = corpus_source or self.pipeline_payload.source

        if corpus_source is None:
            return None

        folder, filename = os.path.split(corpus_source)
        return jj(folder, "shared", "checkpoints", f'{pu.strip_extensions(filename)}_feather')

    def corpus_source_exists(self):
        if self.pipeline_payload.source is None:
            return False
        return os.path.isfile(self.pipeline_payload.source)

    @staticmethod
    def create(
        corpus_name: str = None,
        corpus_type: CorpusType = CorpusType.Undefined,
        corpus_pattern: str = "*.zip",
        serialize_opts: Optional[SerializeOpts] = None,
        text_reader_opts: TextReaderOpts = None,
        text_transform_opts: TextTransformOpts = None,
        pipelines: dict = None,
        pipeline_payload: interfaces.PipelinePayload = None,
        language: str = "english",
        dependencies: dict = None,
        extra_opts: dict = None,
    ) -> CorpusConfig:
        return CorpusConfig(
            corpus_name=corpus_name,
            corpus_type=corpus_type,
            corpus_pattern=corpus_pattern,
            serialize_opts=serialize_opts,
            text_reader_opts=text_reader_opts,
            text_transform_opts=text_transform_opts,
            pipelines=pipelines,
            pipeline_payload=pipeline_payload,
            language=language,
            dependencies=dependencies or {},
            extra_opts=extra_opts or {},
        )

    def resolve_dependency(self, key: str, **kwargs) -> Any:
        """Returns a dependency by key"""
        try:
            if key not in self._key_container:
                self._key_container[key] = DependencyResolver.resolve_key(key, self.dependency_store(), **kwargs)
        except Exception as ex:
            raise DependencyError(f"Dependency {key} not configured.") from ex
        return self._key_container[key]

    def dependency_store(self) -> dict:
        store: dict = {}
        store |= self.dependencies or {}
        for k, v in (self.pipeline_payload.memory_store or {}).items():
            if not isinstance(v, dict):
                continue
            if 'class_name' in v:
                store[k] = v
        return store

    @property
    def pivot_keys(self) -> Any:
        return self.extra_opts.get("pivot_keys")


class DependencyResolver:
    @classmethod
    def resolve_key(cls, key: str, store: dict, **kwargs) -> Any:
        """Returns a dependency by key
        Store is a dictionary of dependencies in the form of:
        key#1: <dependency key>
            class_name: <class name>
            options: <dict of options>
            dependencies: <dict of key-specific dependencies>
        key#2: ...
        """
        dependency: dict = (store or {}).get(key)

        if not dependency:
            raise ValueError(f"Missing dependency in config: {key}")

        class_name: str = dependency.get('class_name')

        if not class_name:
            raise ValueError(f"Missing class name in config: {key}")

        options: dict = dependency.get('options', {}) or {}

        arguments: list = dependency.get('arguments', []) or []
        if not isinstance(arguments, list):
            arguments = [arguments]

        local_store: dict = dict(dependency.get('dependencies', {}) or {})

        cls.resolve_arguments(options, ('config@', store), ('local@', local_store))

        return pu.create_class(class_name)(*arguments, **options, **kwargs)

    @classmethod
    def resolve_arguments(cls, options: dict[str, Any], *stores: list[tuple[str, dict]]):
        for key, value in options.items():
            if not isinstance(value, str):
                continue
            for prefix, store in stores:
                if value.startswith(prefix):
                    options[key] = cls.resolve_key(value.lstrip(prefix), store)
                    break
