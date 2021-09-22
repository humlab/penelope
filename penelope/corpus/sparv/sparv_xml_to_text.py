# -*- coding: utf-8 -*-
import logging
import os

from lxml import etree

from ..readers import ExtractTaggedTokensOpts

logger = logging.getLogger(__name__)

script_path = os.path.dirname(os.path.abspath(__file__))
XSLT_FILENAME = os.path.join(script_path, 'sparv_xml_extract.xslt')
XSLT_FILENAME_V3 = os.path.join(script_path, 'sparv_xml_extract.v3.xslt')

# pylint: disable=too-many-instance-attributes


def snuttify(token):
    if token is None:
        token = ''
    if token.startswith("'") and token.endswith("'"):
        return token
    return "'{}'".format(token)


# FIXME Implement pos_paddings
class SparvXml2Text:
    def __init__(
        self,
        xslt_filename: str = None,
        extract_tokens_opts: ExtractTaggedTokensOpts = None,
        delimiter: str = " ",
    ):
        self.extract_tokens_opts = extract_tokens_opts or ExtractTaggedTokensOpts(lemmatize=True)
        self.xslt_filename = xslt_filename or XSLT_FILENAME
        self.xslt = etree.parse(self.xslt_filename)  # pylint: disable=I1101
        self.xslt_transformer = etree.XSLT(self.xslt)  # pylint: disable=I1101
        self.delimiter = snuttify(delimiter)

        if len(self.extract_tokens_opts.get_passthrough_tokens()) > 0:
            raise ValueError("use of passthrough not implemented for Sparv XML files")

    def transform(self, content):
        xml = etree.XML(content)  # pylint: disable=I1101
        return self._transform(xml)

    def read_transform(self, filename):
        xml = etree.parse(filename)  # pylint: disable=I1101
        return self._transform(xml)

    def _transform(self, xml):
        _opts = self.extract_tokens_opts
        _target = "'lemma'" if _opts.lemmatize is True else "'content'"
        _pos_includes = snuttify(_opts.pos_includes or "")
        _pos_excludes = snuttify(_opts.pos_excludes or "")
        # _pos_paddings = snuttify(_opts.pos_paddings or "")
        _append_pos = snuttify("|" if _opts.append_pos else "")
        text = self.xslt_transformer(
            xml,
            pos_includes=_pos_includes,
            # pos_paddings=_pos_paddings,
            delimiter=self.delimiter,
            target=_target,
            append_pos=_append_pos,
            pos_excludes=_pos_excludes,
        )
        return str(text)
