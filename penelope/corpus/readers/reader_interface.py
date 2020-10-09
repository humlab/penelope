
from abc import ABCMeta, abstractmethod


class ITextReader(metaclass=ABCMeta):

    @abstractmethod
    def __iter__(self):
        while False:
            yield None

    @property
    @abstractmethod
    def metadata(self):
        pass

    @property
    @abstractmethod
    def filenames(self):
        pass
