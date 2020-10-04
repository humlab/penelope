# -*- coding: utf-8 -*-
import logging
import os

from lxml import etree

logger = logging.getLogger(__name__)

script_path = os.path.dirname(os.path.abspath(__file__))
XSLT_FILENAME = os.path.join(script_path, 'sparv_xml_extract.xslt')
XSLT_FILENAME_V3 = os.path.join(script_path, 'sparv_xml_extract.v3.xslt')

# pylint: disable=too-many-instance-attributes


class SparvXml2Text:
    def __init__(
        self,
        xslt_filename=None,
        pos_includes=None,
        lemmatize=True,
        delimiter=" ",
        append_pos="",
        pos_excludes="|MAD|MID|PAD|",
    ):

        self.xslt_filename = xslt_filename or XSLT_FILENAME
        self.pos_includes = self.snuttify(pos_includes) if pos_includes is not None else ''
        self.xslt = etree.parse(self.xslt_filename)  # pylint: disable=I1101
        self.xslt_transformer = etree.XSLT(self.xslt)  # pylint: disable=I1101
        self.delimiter = self.snuttify(delimiter)
        self.lemmatize = lemmatize
        self.pos_excludes = self.snuttify(pos_excludes)
        self.append_pos = self.snuttify(append_pos)
        self.target = "'lemma'" if self.lemmatize is True else "'content'"

    def transform(self, content):
        xml = etree.XML(content)  # pylint: disable=I1101
        return self._transform(xml)

    def read_transform(self, filename):
        xml = etree.parse(filename)  # pylint: disable=I1101
        return self._transform(xml)

    def snuttify(self, token):
        if token.startswith("'") and token.endswith("'"):
            return token
        return "'{}'".format(token)

    def _transform(self, xml):
        text = self.xslt_transformer(
            xml,
            pos_includes=self.pos_includes,
            delimiter=self.delimiter,
            target=self.target,
            append_pos=self.append_pos,
            pos_excludes=self.pos_excludes,
        )
        return str(text)
