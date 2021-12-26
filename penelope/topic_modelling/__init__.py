# type: ignore

from . import prevelance
from .engines import (
    EngineKey,
    EngineSpec,
    get_engine_by_model_type,
    get_engine_cls_by_method_name,
    get_engine_module_by_method_name,
)
from .helper import FilterDocumentTopicWeights
from .interfaces import InferredModel, InferredTopicsData, TrainingCorpus
from .predict import predict_topics
from .prevelance import (
    AverageTopicPrevalenceOverTimeCalculator,
    MemoizedTopicPrevalenceOverTimeCalculator,
    RollingAverageTopicPrevalenceOverTimeCalculator,
    TopicPrevalenceOverTimeCalculator,
    compute_yearly_topic_weights,
)
from .saliency import compute_KL_divergence, compute_term_frequency, compute_term_info, compute_topic_metrics
from .train import train_model
from .utility import (
    compute_topic_proportions,
    filter_document_topic_weights,
    filter_topic_tokens_overview,
    find_inferred_topics_folders,
    find_models,
    get_relevant_topic_documents,
    get_topic_title,
    get_topic_title2,
    get_topic_titles,
    get_topic_top_tokens,
    top_topic_token_weights,
)

YEARLY_AVERAGE_COMPUTE_METHODS = [
    {'key': 'max_weight', 'description': 'Max value', 'tooltip': 'Use maximum value over documents'},
    {
        'key': 'average_weight',
        'description': 'Average of document-topic weights filtered by threshold',
        'tooltip': 'Use average of document where topic is returned by engine (Gensim skips weights less than 1%), above 0 or above given threshold',
    },
    {
        'key': 'true_average_weight',
        'description': 'Average of all documents, even those where weight is 0',
        'tooltip': 'Use average of all document weights even those where topic weight is 0',
    },
]
