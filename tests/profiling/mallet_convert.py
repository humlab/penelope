from __future__ import annotations

from penelope.topic_modelling.engines.engine_gensim.wrappers.convert import (
    convert_document_topics,
    convert_topic_tokens,
)

if __name__ == '__main__':

    data_folder: str = "/data/westac/riksdagen_corpus_data/tm_1920-2020_500-topics-mallet/mallet"

    # convert_topic_tokens(data_folder)
    convert_document_topics(data_folder, normalize=True, epsilon=0.005)


# # df = pd.read_feather(feather_filename)
