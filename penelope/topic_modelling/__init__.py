# type: ignore

from .engines import get_engine_by_model_type, get_engine_cls_by_method_name, get_engine_module_by_method_name
from .interfaces import InferredModel, InferredTopicsData, TrainingCorpus
from .predict import predict_topics
from .saliency import compute_KL_divergence, compute_term_frequency, compute_term_info, compute_topic_metrics
from .train import train_model
from .utility import (
    DocumentTopicWeights,
    compute_topic_proportions,
    compute_topic_yearly_means,
    filter_document_topic_weights,
    find_inferred_topics_folders,
    find_models,
    get_topic_documents,
    get_topic_title,
    get_topic_title2,
    get_topic_titles,
    get_topic_top_tokens,
    top_topic_token_weights,
    filter_topic_tokens_overview
)

YEARLY_MEAN_COMPUTE_METHODS = [
    {'key': 'max_weight', 'description': 'Max value', 'tooltip': 'Use maximum value over documents'},
    {
        'key': 'false_mean',
        'description': 'Mean where topic is relevant',
        'tooltip': 'Use mean value of all documents where topic is above certain treshold',
    },
    {
        'key': 'true_mean',
        'description': 'Mean of all documents',
        'tooltip': 'Use mean value of all documents even those where topic is zero',
    },
]
