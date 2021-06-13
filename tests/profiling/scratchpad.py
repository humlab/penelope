# %%

import os

# %%
import numpy as np
import pandas as pd
from penelope.co_occurrence import Bundle, load_co_occurrences
from penelope.corpus import DocumentIndex, DocumentIndexHelper, Token2Id
from penelope.corpus.dtm import WORD_PAIR_DELIMITER

jj = os.path.join


filename1 = '/home/roger/source/penelope/tests/test_data/VENUS/VENUS_co-occurrence.csv.zip'
b1: Bundle = Bundle.load(filename1)


filename2 = '/home/roger/source/penelope/tmp/VENUS/VENUS_co-occurrence.csv.zip'
b2: Bundle = Bundle.load(filename2)


# %%
# helper: CoOccurrenceHelper = CoOccurrenceHelper()

cc1 = (
    b1.decoded_co_occurrences.assign(token=b1.decoded_co_occurrences.w1 + "/" + b1.decoded_co_occurrences.w2)
    .groupby('token')['value']
    .sum()
    .sort_values(ascending=False)
)

cc2 = (
    b2.decoded_co_occurrences.assign(token=b2.decoded_co_occurrences.w1 + "/" + b2.decoded_co_occurrences.w2)
    .groupby('token')['value']
    .sum()
    .sort_values(ascending=False)
)


# %%

# %%

DATA_FOLDER = '/home/roger/source/penelope/tests/output/'
CO_OCCURRENCE_FILENAME = 'riksdagens-protokoll.1920-2019.test.sparv4.csv.co-occurrences.feather'

DICTIONARY_FILENAME = 'riksdagens-protokoll.1920-2019.test.sparv4.csv.co-occurrences.dictionary.zip'
DOCUMENT_INDEX_FILENAME = 'riksdagens-protokoll.1920-2019.test.sparv4.csv.co-occurrences.document_index.zip'

co_occurrences: pd.DataFrame = load_co_occurrences(jj(DATA_FOLDER, CO_OCCURRENCE_FILENAME))
token2id: Token2Id = Token2Id.load(jj(DATA_FOLDER, DICTIONARY_FILENAME))
id2token = token2id.id2token
token2id.close()

document_index: DocumentIndex = DocumentIndexHelper.load(jj(DATA_FOLDER, DOCUMENT_INDEX_FILENAME)).document_index

co_occurrences.head()

# %%

to_token = lambda x: token2id.id2token.get(x, '').replace(WORD_PAIR_DELIMITER, '')
token_pairs = co_occurrences[["w1_id", "w2_id"]].drop_duplicates()
token_pairs["token"] = token_pairs.w1_id.apply(to_token) + WORD_PAIR_DELIMITER + token_pairs.w2_id.apply(to_token)
token_pairs["token_id"] = token_pairs.index

vocabulary = token_pairs.set_index("token").to_dict()

# %%
document_index = document_index.set_index('document_id', drop=False).rename_axis('').sort_index()
# %%
len(document_index)
# %%
# %%
token_ids = co_occurrences.merge(
    token_pairs.set_index(['w1_id', 'w2_id']),
    how='left',
    left_on=['w1_id', 'w2_id'],
    right_index=True,
).token_id
# %%
token_ids = (
    co_occurrences.set_index(['w1_id', 'w2_id'])
    .merge(
        token_pairs.set_index(['w1_id', 'w2_id']),
        how='left',
        left_index=True,
        right_index=True,
    )
    .token_id
)

token_ids.astype(np.uint32)
# %%
