import gzip
import logging
import shutil

from penelope.vendor.nltk import sent_tokenize, word_tokenize

from .readers.interfaces import ICorpusReader
from .tokenized_corpus import TokenizedCorpus
from .tokens_transformer import TokensTransformOpts

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


def compress(filename):
    with open(filename, 'rb') as f_in:
        with gzip.open(filename + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


class SentenceSegmenter:
    def __init__(self, content):
        self.content = content

    def __iter__(self):
        for sentence in sent_tokenize(self.content, language='swedish'):
            tokens = word_tokenize(sentence)
            if len(tokens) > 0:
                yield tokens


class DocumentSegmenter:
    def __init__(self, content):
        self.content = content

    def __iter__(self):
        tokens = word_tokenize(self.content)
        if len(tokens) > 0:
            yield tokens


class ChunkSegmenter:
    def __init__(self, content, chunk_size=500):

        assert chunk_size > 0

        self.content = content
        self.chunk_size = chunk_size

    def __iter__(self):
        tokens = word_tokenize(self.content)
        for i in range(0, len(tokens), self.chunk_size):
            yield tokens[i : i + self.chunk_size]


class SegmentedTextCorpus(TokenizedCorpus):
    """
    This is a tokenized corpus in the format [ (doc_1, tokens_1), ....(doc_n, tokens_n)]
    where tokens_x is a sequence of strings. The token order is preserved i.e. it is not a BOW
    """

    def __init__(
        self,
        reader: ICorpusReader,
        segment_strategy: str = 'sentence',
        segment_size: int = 0,
        *,
        tokens_transform_opts: TokensTransformOpts = None,
    ):

        if segment_strategy not in ['sentence', 'chunk', 'document']:
            raise AssertionError('Attribute segment_strategy must be specified for text corpus')

        if segment_strategy == 'chunk' and segment_size <= 0:
            raise AssertionError('Attribute segment_chunk must be positive if segement_strategy is chunk')

        super().__init__(reader, tokens_transform_opts=tokens_transform_opts)
        self.segment_strategy = segment_strategy
        self.segment_size = segment_size
        self.strategies = {
            'sentence': lambda x, _: SentenceSegmenter(x),
            'chunk': ChunkSegmenter,
            'document': lambda x, _: DocumentSegmenter(x),
        }
        self.stats = []

    def get_segmenter(self, content):

        return (self.strategies[self.segment_strategy])(content, self.segment_size)

    @property
    def terms(self):
        return (x[1] for x in self)

    def _create_iterator(self):

        stats = []

        for filename, content in self.reader:
            total_token_count = 0
            token_count = 0
            for i, tokens in enumerate(self.get_segmenter(content)):
                total_token_count += len(tokens)
                tokens = self.transformer.transform(tokens)
                token_count += len(tokens)
                if len(tokens) > 0:
                    yield f"{filename}#{i}", tokens

            stats.append((filename, total_token_count, token_count))

        self.stats = stats
