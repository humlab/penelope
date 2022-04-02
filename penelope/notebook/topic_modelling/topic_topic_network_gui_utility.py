from typing import Tuple

import bokeh
import bokeh.plotting
import pandas as pd
from loguru import logger

import penelope.network.networkx.utility as network_utility
from penelope.network import plot_utility


# pylint: disable=too-many-arguments, too-many-locals
def display_topic_topic_network(
    data: pd.DataFrame,
    layout: str = 'Fruchterman-Reingold',
    scale: float = 1.0,
    element_id: str = '',
    titles=None,
    topic_proportions=None,
    node_range: Tuple[int, int] = (20, 60),
    edge_range: Tuple[int, int] = (1, 10),
):
    if len(data) == 0:
        logger.info('No data. Please change selections.')
        return

    network = network_utility.create_network(
        data,
        source_field='source',
        target_field='target',
        weight='n_docs',
    )
    p = plot_utility.plot_network(
        network=network,
        layout_algorithm=layout,
        scale=scale,
        threshold=0.0,
        node_description=titles,
        node_proportions=topic_proportions,
        weight_scale=1.0,
        normalize_weights=False,
        element_id=element_id,
        figsize=(1200, 800),
        node_range=node_range,
        edge_range=edge_range,
    )
    bokeh.plotting.show(p)
