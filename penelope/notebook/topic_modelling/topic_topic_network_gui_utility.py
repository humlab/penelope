# Visualize topic co-occurrence
from typing import Any, Dict, List, Tuple, Union

import bokeh
import bokeh.plotting
import pandas as pd
from IPython.display import display
from loguru import logger

import penelope.network.networkx.utility as network_utility
from penelope import topic_modelling, utility
from penelope.network import plot_utility


def get_topic_titles(topic_token_weights, topic_id=None, n_words=100):
    df_temp = (
        topic_token_weights if topic_id is None else topic_token_weights[(topic_token_weights.topic_id == topic_id)]
    )
    df = (
        df_temp.sort_values('weight', ascending=False)
        .groupby('topic_id')
        .apply(lambda x: ' '.join(x.token[:n_words].str.title()))
    )
    return df


def get_filtered_network_data(
    inferred_topics: topic_modelling.InferredTopicsData,
    filters: Dict[str, Any],
    threshold: float,
    ignores: List[int],
    period: Union[int, Tuple[int, int]],
    n_docs: int,
) -> pd.DataFrame:

    dtw: pd.DataFrame = (
        inferred_topics.calculator.reset().filter_by_keys(**filters).threshold(threshold=threshold).value
    )

    if ignores is not None:
        dtw = dtw[~dtw.topic_id.isin(ignores)]

    if len(period or []) == 2:
        dtw = dtw[(dtw.year >= period[0]) & (dtw.year <= period[1])]

    if isinstance(period, int):
        dtw = dtw[dtw.year == period]

    topic_topic: pd.DataFrame = dtw.merge(dtw, how='inner', left_index='document_id', right_on='document_id')

    topic_topic = topic_topic[(topic_topic.topic_id_x < topic_topic.topic_id_y)]

    topic_topic = topic_topic.groupby([topic_topic.topic_id_x, topic_topic.topic_id_y]).size().reset_index()

    topic_topic.columns = ['source', 'target', 'n_docs']

    # FIXME: Måste normalisera efter antal dokument!!!
    if n_docs > 1:
        topic_topic = topic_topic[topic_topic.n_docs >= n_docs]

    return topic_topic


# pylint: disable=too-many-arguments, too-many-locals
def display_topic_topic_network(
    inferred_topics: topic_modelling.InferredTopicsData,
    filters: Dict[str, Any],
    period: Union[int, Tuple[int, int]] = None,
    ignores: List[int] = None,
    threshold: float = 0.10,
    layout: str = 'Fruchterman-Reingold',
    n_docs: int = 1,
    scale: float = 1.0,
    output_format: str = 'table',
    element_id: str = '',
    titles=None,
    topic_proportions=None,
    node_range: Tuple[int, int] = (20, 60),
    edge_range: Tuple[int, int] = (1, 10),
):
    try:

        df = get_filtered_network_data(inferred_topics, filters, threshold, ignores, period, n_docs)

        if len(df) == 0:
            print('No data. Please change selections.')
            return

        if output_format == 'network':
            network = network_utility.create_network(df, source_field='source', target_field='target', weight='n_docs')
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
        else:

            df.columns = ['Source', 'Target', 'DocCount']
            if output_format == 'table':
                display(df)
            elif output_format.lower() in ('xlsx', 'csv', 'clipboard'):
                utility.ts_store(data=df, extension=output_format.lower(), basename='topic_topic_network')

    except Exception as ex:  # pylint: disable=bare-except
        print("No data: please adjust filters")
        logger.info(ex)
