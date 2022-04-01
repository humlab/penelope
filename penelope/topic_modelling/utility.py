import glob
import os
from typing import List

from penelope import utility as pu


def find_models(path: str) -> dict:
    """Return subfolders containing a computed topic model in specified path"""
    folders = [os.path.split(x)[0] for x in glob.glob(os.path.join(path, "**", "model_options.json"), recursive=True)]
    models = [
        {'folder': x, 'name': os.path.split(x)[1], 'options': pu.read_json(os.path.join(x, "model_options.json"))}
        for x in folders
    ]
    return models


def find_model(path: str, model_name: str) -> dict:
    return next((x for x in find_models(path) if x["name"] == model_name), None)


def find_inferred_topics_folders(folder: str) -> List[str]:
    """Return inferred data in sub-folders to `folder`"""
    filenames = glob.glob(os.path.join(folder, "**/*document_topic_weights.zip"), recursive=True)
    folders = [os.path.split(filename)[0] for filename in filenames]
    return folders
