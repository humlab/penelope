# type: ignore
from .interfaces import ExtractTaggedTokensOpts, ICorpusReader, PhraseSubstitutions, TextSource
from .pandas_reader import PandasCorpusReader
from .sparv_csv_tokenizer import SparvCsvTokenizer
from .sparv_xml_tokenizer import Sparv3XmlReader, SparvXmlReader
from .streamify_text_source import streamify_text_source
from .text_reader import TextReader, TextReaderOpts
from .text_tokenizer import TextTokenizer
from .text_transformer import TEXT_TRANSFORMS, TextTransformer, TextTransformOpts
from .zip_iterator import ZipTextIterator
