from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any, Type

from . import engine_gensim  # , engine_textacy
from .engine_gensim.options import EngineKey  # pylint: disable=unused-import
from .interface import EngineSpec  # pylint: disable=unused-import

if TYPE_CHECKING:
    from . import interface


ENGINES = {
    # 'sklearn': engine_textacy,
    'gensim_': engine_gensim,
}


def get_engine_module_by_method_name(method: str) -> ModuleType:
    for key in ENGINES:
        if method.startswith(key):
            return ENGINES[key]
    raise ValueError(f"Unknown method {method}")


def get_engine_cls_by_method_name(method: str) -> Type[interface.ITopicModelEngine]:
    return get_engine_module_by_method_name(method).TopicModelEngine


def get_engine_by_model_type(model: Any) -> interface.ITopicModelEngine:

    for _, engine in ENGINES.items():
        if engine.is_supported(model):
            return engine.TopicModelEngine(model)

    raise ValueError(f"unsupported model {type(model)}")
