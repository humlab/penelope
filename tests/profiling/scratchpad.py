# %%

import os

# %%
import numpy as np
import pandas as pd
from penelope.co_occurrence import load_co_occurrences
from penelope.corpus import DocumentIndex, DocumentIndexHelper, Token2Id

jj = os.path.join


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

to_token = token2id.id2token.get
token_pairs = co_occurrences[["w1_id", "w2_id"]].drop_duplicates()
token_pairs["token"] = token_pairs.w1_id.apply(to_token) + "/" + token_pairs.w2_id.apply(to_token)
token_pairs["token_id"] = token_pairs.index

vocabulary = token_pairs.set_index("token").to_dict()

# %%
document_index = document_index.set_index('document_id', drop=False).rename_axis('').sort_index()
# %%
len(document_index)
# %%
document_index.index.size
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
