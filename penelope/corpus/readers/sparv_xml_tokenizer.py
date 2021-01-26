# -*- coding: utf-8 -*-
import logging

from penelope.corpus.sparv.sparv_xml_to_text import XSLT_FILENAME_V3, SparvXml2Text

from .interfaces import ExtractTaggedTokensOpts, TextReaderOpts, TextSource
from .text_tokenizer import TextTokenizer
from .text_transformer import TextTransformOpts

logger = logging.getLogger(__name__)

# pylint: disable=too-many-arguments, super-with-arguments, too-many-instance-attributes


class SparvXmlReader(TextTokenizer):
    def __init__(
        self,
        source: TextSource,
        *,
        reader_opts: TextReaderOpts = None,
        version: int = 4,
        xslt_filename: str = None,
        extract_tokens_opts: ExtractTaggedTokensOpts = None,
        chunk_size: int = None,
    ):
        """Sparv XML file reader

        Parameters
        ----------
        source : TextSource
            Source (filename, ZIP, tokens_reader)
        extract_tokens_opts : ExtractTaggedTokensOpts, optional
        reader_opts: TextReaderOpts
        chunk_size: int
        xslt_filename : str, optional
            XSLT filename, by default None
        version : int, optional
            Sparv version, by default 4
        """
        reader_opts = (reader_opts or TextReaderOpts()).copy(filename_pattern='*.xml', as_binary=True)
        self.delimiter: str = ' '
        super().__init__(
            source,
            reader_opts=reader_opts,
            transform_opts=TextTransformOpts.empty(),
            tokenize=lambda x: x.split(self.delimiter),
            chunk_size=chunk_size,
        )

        self.extract_tokens_opts = extract_tokens_opts or ExtractTaggedTokensOpts(lemmatize=True)
        self.xslt_filename = XSLT_FILENAME_V3 if version == 3 else xslt_filename
        self.parser = SparvXml2Text(
            xslt_filename=self.xslt_filename,
            delimiter=self.delimiter,
            extract_tokens_opts=self.extract_tokens_opts,
        )

    def preprocess(self, content):
        return self.parser.transform(content)


class Sparv3XmlReader(SparvXmlReader):
    def __init__(
        self,
        source: TextSource,
        *,
        extract_tokens_opts: ExtractTaggedTokensOpts = None,
        reader_opts: TextReaderOpts = None,
        chunk_size: int = None,
    ):
        """Sparv v3 XML file reader

        Parameters
        ----------
        source : TextSource
            Source (filename, folder, zip, list)
        extract_tokens_opts: ExtractTaggedTokensOpts, optional
        reader_opts : TextReaderOpts
        """
        reader_opts = reader_opts or TextReaderOpts()
        super().__init__(
            source,
            extract_tokens_opts=extract_tokens_opts,
            xslt_filename=XSLT_FILENAME_V3,
            reader_opts=reader_opts,
            chunk_size=chunk_size,
        )
