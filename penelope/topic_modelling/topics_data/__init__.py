# type: ignore

from .document import (
    DocumentTopicsCalculator,
    compute_topic_proportions,
    filter_by_data_keys,
    filter_by_document_index_keys,
    filter_by_keys,
    filter_by_n_top,
    filter_by_text,
    filter_by_threshold,
    filter_by_topics,
    filter_topic_tokens_overview,
    overload,
)
from .prevelance import (
    AverageTopicPrevalenceOverTimeCalculator,
    MemoizedTopicPrevalenceOverTimeCalculator,
    TopicPrevalenceOverTimeCalculator,
    compute_yearly_topic_weights,
)
from .token import TopicTokensMixIn, get_topic_title, get_topic_title2, get_topic_titles, top_topic_token_weights
from .topics_data import InferredTopicsData, PickleUtility

YEARLY_AVERAGE_COMPUTE_METHODS = [
    {
        'key': 'max_weight',
        'description': 'Max value',
        'short_description': 'Max value',
        'tooltip': 'Use maximum value over documents',
    },
    {
        'key': 'average_weight',
        'description': 'Average of document-topic weights filtered by threshold',
        'short_description': 'Average above threshold',
        'tooltip': 'Use average of document where topic is returned by engine (Gensim skips weights less than 1%), above 0 or above given threshold',
    },
    {
        'key': 'true_average_weight',
        'description': 'Average of all documents, even those where weight is 0',
        'short_description': 'Average of all weights',
        'tooltip': 'Use average of all document weights even those where topic weight is 0',
    },
]
