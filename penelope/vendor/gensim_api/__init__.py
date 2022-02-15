# type: ignore
# pylint: disable=unused-import

from ._gensim import check_output, corpora, models

try:
    import gensim

    GENSIM_INSTALLED = True
except ImportError:
    GENSIM_INSTALLED = False
