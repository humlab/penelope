import logging
from typing import Any, AnyStr, Callable, Iterable, List, Tuple

import pandas as pd
from gensim.corpora.textcorpus import TextCorpus
from penelope.corpus import DocumentIndex
from penelope.utility import streamify_any_source

logger = logging.getLogger(__name__)


# pylint: disable=abstract-method
class ExtTextCorpus(TextCorpus):
    def __init__(
        self,
        stream: Iterable[Tuple[str, AnyStr]],
        dictionary: dict = None,
        metadata=False,
        character_filters=None,
        tokenizer: Callable[[str], List[str]] = None,
        token_filters=None,
        bigram_transform=False,  # pylint: disable=unused-argument
    ):
        self.stream: Iterable[Tuple[str, AnyStr]] = stream
        self.filenames: List[str] = None
        self.document_index: DocumentIndex = None
        self.length: int = None

        # if 'filenames' in content_iterator.__dict__:
        #    self.filenames = content_iterator.filenames
        #    self.document_names = self._compile_documents()
        #    self.length = len(self.filenames)

        token_filters = self.default_token_filters() + (token_filters or [])

        # if bigram_transform is True:
        #    train_corpus = GenericTextCorpus(content_iterator, token_filters=[ x.lower() for x in tokens ])
        #    phrases = gensim.models.phrases.Phrases(train_corpus)
        #    bigram = gensim.models.phrases.Phraser(phrases)
        #    token_filters.append(
        #        lambda tokens: bigram[tokens]
        #    )

        super().__init__(
            input=True,
            dictionary=dictionary,
            metadata=metadata,
            character_filters=character_filters,
            tokenizer=tokenizer,
            token_filters=token_filters,
        )

    def default_token_filters(self):
        return [
            (lambda tokens: [x.lower() for x in tokens]),
            (lambda tokens: [x for x in tokens if any(map(lambda x: x.isalpha(), x))]),
        ]

    def getstream(self):
        """Generate documents from the underlying plain text collection (of one or more files).
        Yields
        ------
        str
            Document read from plain-text file.
        Notes
        -----
        After generator end - initialize self.length attribute.
        """

        document_infos = []
        for filename, content in self.stream:
            yield content
            document_infos.append({'document_name': filename})

        self.length = len(document_infos)
        self.document_index: DocumentIndex = pd.DataFrame(document_infos)
        self.filenames: List[str] = list(self.document_index.document_name.values)

    def get_texts(self):
        """
        This is mandatory method from gensim.corpora.TextCorpus. Returns stream of documents.
        """
        for document in self.getstream():
            yield self.preprocess_text(document)

    def preprocess_text(self, text) -> List[str]:
        """Apply `self.character_filters`, `self.tokenizer`, `self.token_filters` to a single text document.

        Parameters
        ---------
        text : str
            Document read from plain-text file.

        Returns
        ------
        list of str
            List of tokens extracted from `text`.

        """
        for character_filter in self.character_filters:
            text = character_filter(text)

        tokens: List[str] = self.tokenizer(text)
        for token_filter in self.token_filters:
            tokens = token_filter(tokens)

        return tokens

    def __get_document_info(self, filename):
        return {
            'document_name': filename,
        }

    def ___compile_documents(self):

        document_data = map(self.__get_document_info, self.filenames)

        document_index: DocumentIndex = pd.DataFrame(list(document_data))
        document_index.index.names = ['document_id']

        return document_index


class SimpleExtTextCorpus(ExtTextCorpus):
    """Reads content in stream and returns tokenized text. No other processing."""

    def __init__(self, source: Any, lowercase: bool = False, filename_filter=None):

        self.reader: Iterable[Tuple[str, AnyStr]] = streamify_any_source(source, filename_filter=filename_filter)
        self.filenames: List[str] = self.reader.filenames
        self.lowercase: bool = lowercase

        super().__init__(self.reader)

    def default_token_filters(self) -> List[Callable]:

        token_filters = [
            (lambda tokens: [x.strip('_') for x in tokens]),
        ]

        if self.lowercase:
            token_filters = token_filters + [(lambda tokens: [x.lower() for x in tokens])]

        return token_filters

    def preprocess_text(self, text: str) -> List[str]:
        return self.tokenizer(text)
