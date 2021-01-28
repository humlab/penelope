# type: ignore
from .compute import infer_model
from .container import InferredModel, InferredTopicsData, TrainingCorpus
from .persist import load_model, store_model
from .predict import compile_inferred_topics_data, predict_document_topics
from .saliency import compute_KL_divergence, compute_term_frequency, compute_term_info, compute_topic_metrics
from .utility import (  # compute_topic_proportions2,
    YEARLY_MEAN_COMPUTE_METHODS,
    compute_topic_proportions,
    compute_topic_yearly_means,
    display_termite_plot,
    document_terms_count,
    find_models,
    get_topic_title,
    get_topic_titles,
    get_topic_tokens,
    get_topics_unstacked,
    id2word_to_dataframe,
    malletmodel2ldamodel,
    normalize_weights,
    plot_topic,
)
