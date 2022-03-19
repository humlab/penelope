import numpy as np
import pandas as pd

from penelope.vendor.gensim_api import models as gensim_models


def diagnostics_to_topic_token_weights_data(
    topic_token_diagnostics: pd.DataFrame, n_tokens: int = 200
) -> list[tuple[int, tuple[str, float]]]:
    """Convert dataframe with MALLET tokens diagnostics data to list if (topic-id, list of (token,weight))"""
    ttd: pd.DataFrame = topic_token_diagnostics
    ttd = ttd[(ttd['rank'] <= n_tokens)]
    ttw: pd.Series = ttd.groupby('topic_id')[['token', 'prob']].apply(lambda x: list(zip(x.token, x.prob)))
    data = list(zip(ttw.index, ttw))
    return data


# NOTE gensim 4.0: wrappers.ldamallet.LdaMallet is deprecated/removed in Gensim 4.0
def malletmodel2ldamodel(mallet_model: gensim_models.LdaMallet, gamma_threshold: float = 0.001, iterations: int = 50):
    """Convert :class:`~gensim.models.wrappers.ldamallet.LdaMallet` to :class:`~gensim.models.ldamodel.LdaModel`.
    This works by copying the training model weights (alpha, beta...) from a trained mallet model into the gensim model.
    Parameters
    ----------
    mallet_model : :class:`~gensim.models.wrappers.ldamallet.LdaMallet`
        Trained Mallet model
    gamma_threshold : float, optional
        To be used for inference in the new LdaModel.
    iterations : int, optional
        Number of iterations to be used for inference in the new LdaModel.
    Returns
    -------
    :class:`~gensim.models.ldamodel.LdaModel`
        Gensim native LDA.
    """
    model_gensim: gensim_models.LdaModel = gensim_models.LdaModel(
        id2word=mallet_model.id2word,
        num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha,
        eta=0,
        iterations=iterations,
        gamma_threshold=gamma_threshold,
        dtype=np.float64,  # don't loose precision when converting from MALLET
    )
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim
