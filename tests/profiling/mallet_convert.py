from __future__ import annotations

from penelope.topic_modelling.engines.engine_gensim.wrappers.convert import (
    convert_dictionary,
    convert_document_index,
    convert_document_topics,
    convert_overview,
    convert_topic_tokens,
    to_feather,
)

if __name__ == '__main__':

    folder: str = "./data/tm_1920-2020_500-TF5-MP0.02.500000.lemma.mallet"

    # convert_document_index(folder)
    # convert_dictionary(folder)
    # convert_overview(folder)
    # convert_topic_tokens(folder)
    convert_document_topics(folder, normalize=True, epsilon=0.005)
    to_feather(folder)
