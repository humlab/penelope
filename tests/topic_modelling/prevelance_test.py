from penelope.notebook.topic_modelling import TopicModelContainer
from penelope.notebook.topic_modelling.topic_trends_gui import TopicTrendsGUI


def test_create_gui(state: TopicModelContainer):
    gui: TopicTrendsGUI = TopicTrendsGUI()
    assert gui is not None

    gui = gui.setup(state)
    assert gui is not None

    layout = gui.layout()
    assert layout is not None

    gui.update_handler()


# class TopicPrevalenceOverTimeCalculator(abc.ABC):
#     @abc.abstractmethod
#     def compute(
#         self,
#         *,
#         inferred_topics: InferredTopicsData,
#         filters: dict,
#         threshold: float = 0.0,
#         result_threshold: float = 0.0,
#     ) -> pd.DataFrame:
#         ...


# class MeanTopicPrevalenceOverTimeCalculator(TopicPrevalenceOverTimeCalculator):
#     def compute(
#         self,
#         *,
#         inferred_topics: InferredTopicsData,
#         filters: dict,
#         threshold: float = 0.0,
#         result_threshold: float = 0.0,
#     ) -> pd.DataFrame:
#         document_topic_weights: pd.DataFrame = (
#             DocumentTopicWeightsReducer(inferred_topics).threshold(threshold).filter_by_keys(filters)
#         )
#         return self.compute_yearly_mean_topic_weights(document_topic_weights, threshold=result_threshold)

#     @staticmethod
#     def compute_yearly_mean_topic_weights(
#         document_topic_weights: pd.DataFrame, threshold: float = None
#     ) -> pd.DataFrame:
#         return compute_yearly_mean_topic_weights(document_topic_weights, threshold)


# class RollingMeanTopicPrevalenceOverTimeCalculator(MeanTopicPrevalenceOverTimeCalculator):
#     """Not implemented"""


# class TopTopicPrevalenceOverTimeCalculator(TopicPrevalenceOverTimeCalculator):
#     """Not implemented"""


# class MemoizedTopicPrevalenceOverTimeCalculator:
#     """Proxy calculator that returns last calculation if arguments are the same"""

#     class ArgsMemory(NamedTuple):
#         inferred_topics: InferredTopicsData
#         filters: dict
#         threshold: float = 0.0
#         result_threshold: float = 0.0

#         def validate(
#             self,
#             inferred_topics: InferredTopicsData,
#             filters: dict,
#             threshold: float = 0.0,
#             result_threshold: float = 0.0,
#         ):
#             return (
#                 self.inferred_topics is inferred_topics
#                 and self.filters == filters
#                 and self.threshold == threshold
#                 and self.result_threshold == result_threshold
#             )

#     def __init__(self, calculator: TopicPrevalenceOverTimeCalculator):

#         self.calculator: TopicPrevalenceOverTimeCalculator = calculator or MeanTopicPrevalenceOverTimeCalculator()
#         self.data: pd.DataFrame = None
#         self.args: MemoizedTopicPrevalenceOverTimeCalculator.ArgsMemory = None

#     def compute(
#         self, inferred_topics: InferredTopicsData, filters: dict, threshold: float = 0.0, result_threshold: float = 0.0
#     ) -> pd.DataFrame:

#         if not (self.args and self.args.validate(inferred_topics, filters, threshold, result_threshold)):

#             self.data = self.calculator.compute(inferred_topics, filters, threshold, result_threshold)
#             self.args = MemoizedTopicPrevalenceOverTimeCalculator.ArgsMemory(
#                 inferred_topics=inferred_topics, filters=filters, threshold=threshold, result_threshold=result_threshold
#             )

#         return self.data


# def compute_yearly_mean_topic_weights(document_topic_weights: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
#     """Compute yearly document topic weights"""

#     dtw: pd.DataFrame = document_topic_weights

#     min_year, max_year = (dtw.year.min(), dtw.year.max())

#     """ Create a dataframe with all year-topic combinations as index """
#     cross_iter = itertools.product(range(min_year, max_year + 1), range(0, dtw.topic_id.max() + 1))
#     yearly_weights: pd.DataFrame = pd.DataFrame(list(cross_iter), columns=['year', 'topic_id']).set_index(
#         ['year', 'topic_id']
#     )

#     """ Add the most basic stats """
#     yearly_weights = yearly_weights.join(
#         dtw.groupby(['year', 'topic_id'])['weight'].agg([np.max, np.sum, np.mean, len]), how='left'
#     ).fillna(0)

#     yearly_weights.columns = ['max_weight', 'sum_weight', 'false_mean', 'n_topic_docs']

#     yearly_weights['n_topic_docs'] = yearly_weights.n_topic_docs.astype(np.uint32)

#     if threshold is not None:

#         yearly_weights.drop(columns='false_mean', inplace=True)

#         mean_revelance: pd.DataFrame = (
#             dtw[dtw.weight >= threshold].groupby(['year', 'topic_id'])['weight'].agg([np.mean])
#         )
#         mean_revelance.columns = ['false_mean']

#         yearly_weights = yearly_weights.join(mean_revelance, how='left').fillna(0)

#     """Add document count per year"""
#     doc_counts = dtw.groupby('year').document_id.nunique().rename('n_documents')
#     yearly_weights = yearly_weights.join(doc_counts, how='left').fillna(0)
#     yearly_weights['n_documents'] = yearly_weights.n_documents.astype(np.uint32)

#     """Compute true mean weights"""
#     yearly_weights['true_mean'] = yearly_weights.apply(lambda x: x['sum_weight'] / x['n_documents'], axis=1)

#     return yearly_weights.reset_index()
