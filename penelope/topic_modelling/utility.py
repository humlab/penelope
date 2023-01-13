import glob
from dataclasses import dataclass
from os.path import join, split

from penelope import utility as pu


def _find_model_options_files(path: str) -> list[str]:
    return glob.glob(join(path, "**", "model_options.json"), recursive=True)


def _find_model_folders(path: str) -> list[str]:
    return [split(x)[0] for x in _find_model_options_files(path)]


@dataclass
class ModelFolder:

    folder: str
    name: str
    options: dict

    def to_dict(self):
        return {
            'folder': self.folder,
            'name': self.name,
            'options': self.options,
        }


def find_models(path: str) -> list[ModelFolder]:
    """Return subfolders to path that contain a computed topic model"""
    models = [
        ModelFolder(
            folder=folder,
            name=split(folder)[1],
            options=pu.read_json(join(folder, "model_options.json")),
        )
        for folder in _find_model_folders(path)
    ]
    return models


def find_inferred_topics_folders(folder: str) -> list[str]:
    """Return inferred data in sub-folders to `folder`"""
    filenames = glob.glob(join(folder, "**/*document_topic_weights.zip"), recursive=True)
    folders = [split(filename)[0] for filename in filenames]
    return folders
