# Visualize topic co-occurrence
import bokeh
import bokeh.plotting
import penelope.network.plot_utility as plot_utility
import penelope.network.utility as network_utility
import penelope.utility as utility
from IPython.display import display

from .utility import filter_document_topic_weights

logger = utility.get_logger()


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


# pylint: disable=too-many-arguments, too-many-locals
def display_topic_topic_network(
    inferred_topics,
    filters,
    period=None,
    ignores=None,
    threshold=0.10,
    layout='Fruchterman-Reingold',
    n_docs=1,
    scale=1.0,
    output_format='table',
    text_id='',
    titles=None,
    topic_proportions=None,
    node_range=(20, 60),
    edge_range=(1, 10),
):
    try:

        documents = inferred_topics.documents

        if 'document_id' not in documents.columns:
            documents['document_id'] = documents.index

        df = filter_document_topic_weights(inferred_topics.document_topic_weights, filters=filters, threshold=threshold)

        if ignores is not None:
            df = df[~df.topic_id.isin(ignores)]

        if len(period or []) == 2:
            df = df[(df.year >= period[0]) & (df.year <= period[1])]

        if isinstance(period, int):
            df = df[df.year == period]

        df = df.merge(df, how='inner', left_on='document_id', right_on='document_id')
        df = df[(df.topic_id_x < df.topic_id_y)]

        df = df.groupby([df.topic_id_x, df.topic_id_y]).size().reset_index()

        df.columns = ['source', 'target', 'n_docs']

        if n_docs > 1:
            df = df[df.n_docs >= n_docs]

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
                element_id=text_id,
                figsize=(1200, 800),
                node_range=node_range,
                edge_range=edge_range,
            )
            bokeh.plotting.show(p)
        else:
            df.columns = ['Source', 'Target', 'DocCount']
            if output_format == 'table':
                display(df)
            if output_format == 'excel':
                filename = utility.timestamp("{}_topic_topic_network.xlsx")
                df.to_excel(filename)
                print('Data stored in file {}'.format(filename))
            if output_format == 'csv':
                filename = utility.timestamp("{}_topic_topic_network.csv")
                df.to_csv(filename, sep='\t')
                print('Data stored in file {}'.format(filename))

    except:  # pylint: disable=bare-except
        print("No data: please adjust filters")
        # raise
