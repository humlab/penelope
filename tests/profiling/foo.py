from penelope import corpus as pc
from penelope import topic_modelling as tm
from gensim.models import LdaMulticore
from penelope.corpus import dtm
from penelope.topic_modelling.predict import to_dataframe
import pandas as pd
import numpy as np
from tqdm import tqdm

model: LdaMulticore = LdaMulticore.load(fname='data/tm_1920-2020_500-topics/gensim.model.gz')

corpus: pc.VectorizedCorpus = pc.VectorizedCorpus.load(tag='train', folder='data/tm_1920-2020_500-topics')
sparse = dtm.to_sparse2corpus(corpus)

engine = tm.get_engine_by_model_type(model)

result = model.get_document_topics(bow=sparse, minimum_probability=0.04)

n = 0
for doc_topics in tqdm(result, miniters=100):
    n += len(doc_topics)
# dtw_iter = engine.predict(topic_model=model, corpus=corpus, minimum_probability=0.04)


# document_topic_weights = pd.DataFrame(dtw_iter, columns=['document_id', 'topic_id', 'weight'])
# document_topic_weights['document_id'] = document_topic_weights.document_id.astype(np.uint32)
# document_topic_weights['topic_id'] = document_topic_weights.topic_id.astype(np.uint16)
# document_topic_weights = pc.DocumentIndexHelper(corpus.document_index).overload(document_topic_weights, 'year')

# #document_topic_weights = to_dataframe(corpus.document_index, dtw_iter)

# topic_token_weights: pd.DataFrame = engine.get_topic_token_weights(vocabulary=corpus.id2token, n_tokens=200)
# topic_token_overview: pd.DataFrame = engine.get_topic_token_overview(topic_token_weights, n_tokens=200)
# document_index: pd.DataFrame = pc.update_document_index_token_counts_by_corpus(corpus.document_index, corpus)


# # topics_data: tm.InferredTopicsData = tm.InferredTopicsData(
# #     dictionary=pc.Token2Id.id2token_to_dataframe(corpus.id2token),
# #     topic_token_weights=topic_token_weights,
# #     topic_token_overview=topic_token_overview,
# #     document_index=document_index,
# #     document_topic_weights=document_topic_weights,
# # )