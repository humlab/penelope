from typing import Any

from lxml import etree

from .sparv_xml_to_text import snuttify

XSLT_DOCUMENT = """<?xml version="1.0"?>

<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" >
<xsl:output method="text" encoding="utf-8"/>

<xsl:strip-space elements="*" />

<xsl:param name="delimiter"/>

<xsl:template match="{}">

    <xsl:variable name="baseform" select="{}"/>
    <xsl:variable name="lemma" select="substring-before(substring-after($baseform,'|'),'|')"/>

    <xsl:value-of select="text()"/>
    <xsl:value-of select="$delimiter" disable-output-escaping="yes"/>
    <xsl:value-of select="$lemma"/>
    <xsl:value-of select="$delimiter" disable-output-escaping="yes"/>
    <xsl:value-of select="@pos"/>
    <xsl:text>&#xd;</xsl:text>

</xsl:template>

</xsl:stylesheet>
"""


class SparvXml2CSV:
    def __init__(self, delimiter: str = '\t', version: int = 4):

        args = (
            (
                "token",
                "@baseform",
            )
            if version == 4
            else (
                "w",
                "@lemma",
            )
        )

        self.xslt = etree.XML(XSLT_DOCUMENT.format(*args))  # pylint: disable=I1101
        self.xslt_transformer = etree.XSLT(self.xslt)  # pylint: disable=I1101
        self.delimiter: str = snuttify(delimiter)

    def transform(self, content: bytes) -> str:
        xml: Any = etree.XML(content)  # pylint: disable=I1101
        return self._transform(xml)

    def _transform(self, xml) -> str:
        text: bytes = self.xslt_transformer(xml, delimiter=self.delimiter)
        return str(text)
