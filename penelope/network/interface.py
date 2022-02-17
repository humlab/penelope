from dataclasses import dataclass, field
from typing import Any, Callable, Tuple

from .networkx.networkx_api import nx


def noop(**_):
    return {}


@dataclass
class LayoutAlgorithm:
    key: str
    package: str
    name: str
    layout_network: Callable[..., Tuple[Any, Any]]

    engine: Any = None
    layout_function: Any = nx.nx_pydot.pydot_layout
    layout_args: Callable = field(default=noop)
