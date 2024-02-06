import gzip
import pickle
from collections import defaultdict
from os.path import isfile


def store_token_set(tokens: set[str], filename: str) -> None:
    with gzip.open(filename, 'wb') as f:
        f.write('\n'.join(list(tokens)).encode())


def store_dict(data: dict, filename: str) -> None:
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)


def load_token_set(filename: str) -> set[str]:
    """Load tokens from `filename`, one token per line"""
    if isfile(filename):
        with gzip.open(filename, 'rb') as f:
            return set(f.read().decode().split('\n'))
    return set()


def load_dict(filename: str) -> defaultdict(int):
    # logger.info(f"loading {filename}")
    if isfile(filename):
        with open(filename, 'rb') as fp:
            return pickle.load(fp)
    return defaultdict(int)
