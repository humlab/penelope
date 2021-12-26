from typing import Any, Dict

import penelope.vendor.gensim as gensim_utility
from gensim.models.coherencemodel import CoherenceModel
from loguru import logger

from . import options


def compute_score(id2word, model, corpus) -> float:
    try:
        dictionary = gensim_utility.from_id2token_to_dictionary(id2word)
        coherence_model = CoherenceModel(model=model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        return coherence_model.get_coherence()
    except Exception as ex:
        logger.error(ex)
    return None


def compute_scores(
    engine_key: options.EngineKey,
    id2word: Dict[int, str],
    corpus: Any,
    start=10,
    stop: int = 20,
    step: int = 10,
    engine_args: Dict[str, Any] = None,
) -> Dict[str, Any]:

    metrics = []

    dictionary = gensim_utility.from_id2token_to_dictionary(id2word)

    for num_topics in range(start, stop, step):

        engine_spec: options.EngineSpec = options.get_engine_specification(engine_key=engine_key)

        model = engine_spec.engine(**engine_spec.get_options(corpus=corpus, id2word=id2word, engine_args=engine_args))

        coherence_score = CoherenceModel(model=model, corpus=corpus, dictionary=dictionary, coherence='u_mass')

        perplexity_score = 2 ** model.log_perplexity(corpus, len(corpus))

        metric = dict(num_topics=num_topics, coherence_score=coherence_score, perplexity_score=perplexity_score)
        metrics.append(metric)

    # filename = os.path.join(target_folder, "metric_scores.json")

    # with open(filename, 'w') as fp:
    #     json.dump(model_data.options, fp)

    return metrics


# Can take a long time to run.
# model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)
# # Show graph
# limit=40; start=2; step=6;
# x = range(start, limit, step)
# plt.plot(x, coherence_values)
# plt.xlabel("Num Topics")
# plt.ylabel("Coherence score")
# plt.legend(("coherence_values"), loc='best')
# plt.show()
