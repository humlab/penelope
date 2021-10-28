from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

from . import engine_gensim, engine_textacy
from .interface import ITopicModelEngine  # type: ignore

if TYPE_CHECKING:
    from . import interface


ENGINES = {
    'sklearn': engine_textacy,
    'gensim_': engine_gensim,
}


def get_engine_module_by_method_name(method: str) -> types.ModuleType:
    for key in ENGINES:
        if method.startswith(key):
            return ENGINES[key]
    raise ValueError(f"Unknown method {method}")


def get_engine_cls_by_method_name(method: str) -> types.Type[interface.ITopicModelEngine]:
    return get_engine_module_by_method_name(method).TopicModelEngine


def get_engine_by_model_type(model: Any) -> interface.ITopicModelEngine:

    for _, engine in ENGINES.items():
        if engine.is_supported(model):
            return engine.TopicModelEngine(model)

    raise ValueError(f"unsupported model {type(model)}")
