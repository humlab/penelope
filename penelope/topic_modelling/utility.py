from dataclasses import dataclass
from glob import glob
from os.path import join, split, splitext

from penelope import utility as pu


def _find_model_options_files(path: str) -> list[str]:
    return glob(join(path, "**", "model_options.json"), recursive=True)


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
    """Return sub-folders to `folder` that contain inferred topics data"""
    folders = sorted(
        list(
            {
                split(p)[0]
                for p in glob(join(folder, "**/document_topic_weights.*"), recursive=True)
                if splitext(p)[1] in ('.zip', '.feather')
            }
        )
    )

    return folders
