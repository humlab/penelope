from typing import Any, Generic, TypeVar

F = TypeVar("F")


class Registry(Generic[F]):
    _items: dict[str, F] = {}

    @classmethod
    def add(cls, fn: F, key: Any = None) -> F:
        """Add item to registry"""
        if key is None:
            key = fn.__name__
        cls._items[key] = fn
        return fn

    @classmethod
    def get(cls, key: Any) -> F:
        """Get item from registry"""
        if key in cls._items:
            return cls._items.get(key)

        raise ValueError(f"{key} is not registered")

    @classmethod
    def __contains__(cls, key: Any) -> bool:
        return key in cls._items

    @classmethod
    def registered(cls, key: Any) -> bool:
        return key in cls._items

    @classmethod
    def register(cls, key: Any = None, kind: str = None, **args):
        def decorator(fn):
            if kind == "function":
                fn = fn(**args)
            cls.add(fn, key)
            return fn

        return decorator
