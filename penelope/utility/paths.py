import os
import sys
from os.path import abspath, isdir, join

from loguru import logger

# project_name: str = 'welfare_state_analytics'
# project_short_name: str = "westac"


def find_ancestor_folder(x: str) -> str:
    return join(os.getcwd().split(x)[0], x)


def find_root_folder(*, project_name: str) -> str:
    if os.environ.get("JUPYTER_IMAGE_SPEC", "") != "":
        return f"/home/jovyan/work/{project_name}"
    folder = find_ancestor_folder(project_name)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    return folder


def find_data_folder(*, project_name: str, project_short_name: str) -> str:

    probe_names: str = [
        f'/data/{project_short_name}',
        f'/data/staging-{project_short_name}',
        f"/home/jovyan/work/{project_name}",
        join(abspath(os.getcwd()), 'data'),
    ]

    for folder in probe_names:
        if isdir(folder):
            return folder
    logger.warning(f"no default data folder found! probed: {', '.join(probe_names)}")
    return '/data'


def find_resources_folder(*, project_name: str, project_short_name: str) -> str:  # pylint: disable=unused-argument
    root_folder: str = find_root_folder(project_name=project_name)
    folder: str = join(root_folder, "resources")
    return folder
