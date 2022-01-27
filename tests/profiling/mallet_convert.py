from __future__ import annotations

import pandas as pd

from penelope.topic_modelling.engines.engine_gensim.wrappers.convert import (
    convert_dictionary,
    convert_document_index,
    convert_document_topics,
    convert_topic_tokens,
)

if __name__ == '__main__':

    data_folder: str = "./data/tm_1920-2020_500-topics-mallet"
    mallet_data_folder: str = "./data/tm_1920-2020_500-topics-mallet/mallet"

    convert_document_index(data_folder)
    convert_dictionary(data_folder)

    convert_topic_tokens(mallet_data_folder)
    convert_document_topics(mallet_data_folder, normalize=True, epsilon=0.005)
