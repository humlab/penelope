from __future__ import annotations

from os.path import isfile
from os.path import join as jj

import numpy as np
import pandas as pd
from smart_open import open as smart_open
from tqdm import tqdm

from penelope import corpus as pc
from penelope import topic_modelling as tm

# pylint: disable=no-member


def probe_filenames(folder: str, filenames: list[str]) -> str | None:
    for filename in filenames:
        if isfile(jj(folder, filename)):
            return jj(folder, filename)
    return None


def convert_topic_tokens(folder: str, source_filename: str = None) -> pd.DataFrame:

    mallet_folder: str = jj(folder, "mallet")

    target_filename = jj(folder, 'topic_token_weights.zip')

    if isfile(target_filename):
        return None

    source_filename: str = source_filename or probe_filenames(
        mallet_folder, ["topicwordweights.txt.gz", "topicwordweights.txt", "topicwordweights.zip"]
    )

    id2token: dict[int, str] = pd.read_json(jj(folder, "topic_model_id2token.json.gz"), typ="series")
    token2id: dict[str, int] = {v: k for k, v in id2token.items()}

    ttw: pd.DataFrame = pd.read_csv(
        source_filename,
        names=['topic_id', 'token', 'weight'],
        dtype={'topic_id': np.int16, 'weight': np.float64},
        header=None,
        sep='\t',
    )

    ttw = ttw[ttw.weight > ttw.weight.min()]  # pylint: disable=no-member

    ttw['token_id'] = ttw.token.apply(token2id.get)

    ttw.drop(columns='token', inplace=True)

    ttw['topic_id'] = ttw.topic_id.astype(np.int16)
    ttw['token_id'] = ttw.token_id.astype(np.int32)

    # df['weight'] = df.weight.astype(np.float32)

    ttw = ttw[['topic_id', 'token_id', 'weight']].reset_index(drop=True)

    # ttw.to_feather(jj(folder, "topic_token_weights.feather"))

    ttw.to_csv(
        target_filename,
        sep='\t',
        compression=dict(method='zip', archive_name="topic_token_weights.csv"),
        header=True,
    )
    return ttw


def convert_overview(folder: str) -> None:

    target_filename: str = jj(folder, 'topic_token_overview.zip')
    if isfile(target_filename):
        return

    source_filename: str = probe_filenames(jj(folder, "mallet"), ["topickeys.txt.gz", "topickeys.txt", "topickeys.zip"])

    df: pd.DataFrame = pd.read_csv(source_filename, sep='\t', names=['topic_id', 'alpha', 'tokens']).set_index(
        'topic_id'
    )
    df.to_csv(
        target_filename,
        sep='\t',
        compression=dict(method='zip', archive_name="topic_token_overview.csv"),
        header=True,
    )
    # df.to_feather(jj(folder, 'topic_token_overview.feather'))


def doctopics_to_dataframe(source_filename: str, normalize: bool = False, epsilon: float = 0.005) -> pd.DataFrame:

    ds, ts, ws = [], [], []

    with smart_open(source_filename, mode='rt') as fp:

        for row in tqdm(fp, mininterval=1.0):

            if row[0] == '#':
                continue

            values: list[float] = np.array(list(float(x) for x in row.strip().split('\t')))

            document_id: int = int(values[0])
            topic_ids: np.ndarray = values[2::2]
            weights: np.ndarray = values[3::2]

            keep_ids: np.ndarray = np.argwhere(weights >= epsilon).T[0]

            topic_ids: np.ndarray = topic_ids[keep_ids].astype(int)
            weights: np.ndarray = weights[keep_ids]

            if normalize and len(weights) > 0:
                weights /= weights.sum()

            ds.append((document_id, len(topic_ids)))
            ts.append(topic_ids)
            ws.append(weights)

    dtw: pd.DataFrame = pd.DataFrame(
        data={
            'document_id': (t for tx in ([d] * n for d, n in ds) for t in tx),
            'topic_id': (t for tx in ts for t in tx),
            'weight': (w for wx in ws for w in wx),
        },
    )
    dtw['topic_id'] = dtw.topic_id.astype(np.int16)
    dtw['document_id'] = dtw.document_id.astype(np.int32)
    return dtw


def convert_document_topics(
    folder: str, source_filename: str = None, normalize: bool = True, epsilon: float = 0.005
) -> pd.DataFrame:
    """Converts a 2.0.8+ MALLET doc-topics file into data frame stored in FEATHER format."""
    mallet_folder: str = jj(folder, "mallet")
    target_filename: str = jj(folder, 'document_topic_weights.zip')

    if isfile(target_filename):
        return

    source_filename: str = source_filename or probe_filenames(
        mallet_folder,
        [
            "doctopics.txt.infer.gz",
            "doctopics.txt.infer.zip",
            "doctopics.txt.infer",
            "doctopics.txt.gz",
            "doctopics.txt.zip",
            "doctopics.zip",
            "doctopics.txt",
        ],
    )
    dtw: pd.DataFrame = doctopics_to_dataframe(source_filename, normalize, epsilon)

    dtw.to_feather(jj(mallet_folder, "doctopics.feather"))

    di: pd.DataFrame = pd.read_csv(jj(folder, "documents.zip"), sep='\t').set_index('document_id', drop=False)
    dtw = dtw.merge(di[['year']], left_on='document_id', right_index=True, how='left')
    dtw['year'] = dtw.year.astype(np.int16)

    dtw.to_csv(
        target_filename,
        sep='\t',
        compression=dict(method='zip', archive_name="document_topic_weights.csv"),
        header=True,
    )

    # dtw.to_feather(jj(folder, 'document_topic_weights.feather'))


def explode_pickle(folder: str) -> None:
    filename: str = jj(folder, 'train_document_index.csv.gz')
    if not isfile(jj(folder, filename)):
        return
    corpus: pc.VectorizedCorpus = pc.VectorizedCorpus.load(folder=folder, tag='train')
    corpus.store_metadata(tag='train', folder=folder, mode='files')


def convert_document_index(folder: str) -> None:

    target_filename: str = jj(folder, "documents.zip")
    source_filename: str = jj(folder, "train_document_index.csv.gz")

    if isfile(target_filename):
        return

    explode_pickle(folder)

    di: pd.DataFrame = (
        pd.read_csv(source_filename, sep=';', index_col=0).set_index('document_name', drop=False).rename_axis('')
    )

    di.to_csv(target_filename, sep='\t', compression=dict(method='zip', archive_name="document_index.csv"), header=True)


def convert_dictionary(folder: str) -> None:

    target_filename: str = jj(folder, "dictionary.zip")
    source_filename: str = jj(folder, "train_token2id.json.gz")

    if isfile(target_filename):
        return

    explode_pickle(folder)

    token2id: pd.Series = pd.read_json(source_filename, typ='series')

    dictionary = pd.DataFrame(data=dict(token_id=token2id, token=token2id.index, dfs=0)).set_index('token_id')

    dictionary.to_csv(
        target_filename, sep='\t', compression=dict(method='zip', archive_name="dictionary.csv"), header=True
    )


def to_feather(folder: str):

    explode_pickle(folder)

    data: tm.InferredTopicsData = tm.InferredTopicsData.load(folder=folder, slim=True)
    data._store_feather(folder)  # pylint: disable=protected-access
