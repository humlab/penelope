from dataclasses import dataclass
from typing import Any, Callable, Tuple

import networkx as nx


@dataclass
class LayoutAlgorithm:
    key: str
    package: str
    name: str
    layout_network: Callable[..., Tuple[Any, Any]]

    engine: Any = None
    layout_function: Any = nx.nx_pydot.pydot_layout
    layout_args: Callable = lambda **_: {}
