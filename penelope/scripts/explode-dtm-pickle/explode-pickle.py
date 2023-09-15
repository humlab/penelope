"""Use this script to explode a pickled DTM saved in pandas="<2.0.0" into separate files.
pickle.load with pandas >= 2.0.0 fails when reading a dict that contains a dataframe stored in pandas < 2.0.0

NOTE! This script must be run with "pandas<2.0.0"!
"""

import fnmatch
import gzip
import json
import os
import pickle
from collections import defaultdict
from os.path import basename, dirname, isfile
from os.path import join as jj

import click
import numpy as np
from loguru import logger


def explode_pickle(*, tag: str, folder: str) -> None:
    """Loads metadata from disk."""
    if isfile(jj(folder, f"{tag}_document_index.csv.gz")):
        logger.info(f"DTM in {folder} with tag {tag} is already exploded")
        return

    pickle_filename: str = jj(folder, f"{tag}_vectorizer_data.pickle")
    if not isfile(pickle_filename):
        logger.info(f"no pickled DTM found in {folder} with tag {tag} (expected filename {pickle_filename})")

    with open(pickle_filename, 'rb') as f:
        data = pickle.load(f)

    store_files(tag=tag, folder=folder, **data)


def store_files(*, tag: str, folder: str, **data) -> None:
    """Stores metadata to disk."""
    if isinstance(data.get('token2id'), defaultdict):
        data['token2id'] = dict(data.get('token2id'))

    data.get('document_index').to_csv(jj(folder, f"{tag}_document_index.csv.gz"), sep=';', compression="gzip")

    with gzip.open(jj(folder, f"{tag}_token2id.json.gz"), 'w') as fp:  # 4. fewer bytes (i.e. gzip)
        fp.write(json.dumps(data.get('token2id')).encode('utf-8'))

    term_frequency: np.ndarray = data.get('overridden_term_frequency')
    if term_frequency is not None:
        np.save(jj(folder, f"{tag}_overridden_term_frequency.npy"), term_frequency, allow_pickle=True)


def find_files(pattern: str, root_path: str):
    for root, dirs, files in os.walk(root_path):
        # remove symbolic link directories from dirs
        dirs[:] = [d for d in dirs if not os.path.islink(jj(root, d))]
        for filename in fnmatch.filter(files, pattern):
            yield jj(root, filename)


@click.command()
@click.argument('root_folder')
def main(root_folder: str = None):
    """
    Explodes pickled DTM files in ROOT_FOLDER recursively.
        input:
            <tag>_vectorizer_data.pickle
        output:
            <tag>_document_index.csv.gz
            <tag>_token2id.json.gz

    The script is intended to be used to explode pickled DTMs saved with pandas < 2.0.0
    Pandas >= 2.0.0 has a breaking change that makes it impossible to read a
    dict that contains a dataframe created with pandas<2.0.0.

    INSTALLATION:
        python -m venv .venv
        . .venv\bin\activate
        pip install -r requirements.txt
    USAGE:

        PYTHONPATH=. python penelope/scripts/explode-dtm-pickle/explode-pickle.py <root_folder>

    """
    logger.info(f"folder: {root_folder}")
    filenames: list[str] = list(find_files("*_vectorizer_data.pickle", root_folder))

    for filename in filenames:
        tag: str = basename(filename)[: -len("_vectorizer_data.pickle")]
        folder = dirname(filename)
        try:
            explode_pickle(tag=tag, folder=folder)
            assert isfile(jj(folder, f"{tag}_document_index.csv.gz"))
            assert isfile(jj(folder, f"{tag}_token2id.json.gz"))
            logger.info(f"{filename} exploded.")
        except Exception as ex:
            logger.error(f"{filename} failed! {str(ex)}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
