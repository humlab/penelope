# type: ignore
# pylint: disable=unused-import
# flake8: noqa

from loguru import logger

from penelope.utility.utils import DummyClass


class CoherenceModel(DummyClass):
    ...


class LdaModel(DummyClass):
    ...


class LdaMulticore(DummyClass):
    ...


class LsiModel(DummyClass):
    ...


class LdaMallet(DummyClass):
    ...


class STTMTopicModel(DummyClass):
    ...


class LdaSeqModel(DummyClass):
    ...


class MalletTopicModel(DummyClass):
    ...


try:

    from gensim.models import CoherenceModel
    from gensim.models.ldamodel import LdaModel
    from gensim.models.ldamulticore import LdaMulticore
    from gensim.models.ldaseqmodel import LdaSeqModel
    from gensim.models.lsimodel import LsiModel

    from .wrappers import LdaMallet, MalletTopicModel, STTMTopicModel

except (ImportError, NameError):
    logger.info("gensim not included in current installment")
