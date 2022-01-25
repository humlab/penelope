from __future__ import annotations

from os.path import join as jj
from smart_open import open

import numpy as np
import pandas as pd
from tqdm import tqdm


def convert_topic_tokens(folder: str, filename: str = "topicwordweights.txt.gz") -> pd.DataFrame:

    filename: str = jj(folder, filename)
    feather_filename: str = jj(folder, "topicwordweights.feather")

    id2token: dict[int, str] = pd.read_json(jj(folder, "..", "topic_model_id2token.json.gz"), typ="series")
    token2id: dict[str, int] = {v: k for k, v in id2token.items()}

    df: pd.DataFrame = pd.read_csv(
        filename,
        names=['topic_id', 'token', 'weight'],
        dtype={'topic_id': np.int16, 'weight': np.float64},
        header=None,
        sep='\t',
    )

    df = df[df.weight > df.weight.min()]

    df['token_id'] = df.token.apply(token2id.get)

    df.drop(columns='token', inplace=True)

    df['topic_id'] = df.topic_id.astype(np.int16)
    df['token_id'] = df.token_id.astype(np.int32)

    # df['weight'] = df.weight.astype(np.float32)

    df = df.reset_index().to_feather(feather_filename)

    return df


def convert_document_topics(
    folder: str, filename: str = "doctopics.txt.infer.gz", normalize: bool = True, epsilon: float = 0.005
) -> pd.DataFrame:
    """Converts a 2.0.8+ MALLET doc-topics file into data frame stored in FEATHER format."""
    filename: str = jj(folder, filename)
    feather_filename: str = jj(folder, "doctopics.feather")

    ds, ts, ws = [], [], []
    with open(filename, mode='rt') as fp:
        for row in tqdm(fp, mininterval=1.0):
            if row[0] == '#':
                continue
            values: list[float] = np.array(list(float(x) for x in row.split('\t')))
            document_id: int = int(values[0])
            topic_weights: np.ndarray = values[2:]
            token_ids: np.ndarray = np.argwhere(topic_weights >= epsilon).T[0]
            weights: np.ndarray = topic_weights[token_ids]
            if normalize and len(weights) > 0:
                weights /= weights.sum()
            ds.append((document_id, len(token_ids)))
            ts.append(token_ids)
            ws.append(topic_weights[token_ids])
    df: pd.DataFrame = pd.DataFrame(
        data={
            'document_id': (t for tx in ([d] * n for d, n in ds) for t in tx),
            'topic_id': (t for tx in ts for t in tx),
            'weight': (w for wx in ws for w in wx),
        },
    )
    df['topic_id'] = df.topic_id.astype(np.int16)
    df['document_id'] = df.document_id.astype(np.int32)

    df.to_feather(feather_filename)
