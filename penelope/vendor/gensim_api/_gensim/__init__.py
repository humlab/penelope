# type: ignore
# flake8: noqa

from subprocess import check_output

from . import _corpus as corpora
from . import _models as models

try:
    from gensim.utils import check_output  # pylint: disable=reimported
except (ImportError, NameError):
    ...
