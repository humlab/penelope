<?xml version="1.0"?>

<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" >
<xsl:output method="text" encoding="utf-8"/>

<xsl:strip-space elements="*" />

<xsl:param name="delimiter"/>

<xsl:template match="token">

    <xsl:variable name="baseform" select="@baseform"/>
    <xsl:variable name="lemma" select="substring-before(substring-after($baseform,'|'),'|')"/>

    <xsl:value-of select="text()"/>
    <xsl:value-of select="$delimiter" disable-output-escaping="yes"/>
    <xsl:value-of select="$lemma"/>
    <xsl:value-of select="$delimiter" disable-output-escaping="yes"/>
    <xsl:value-of select="@pos"/>
    <xsl:text>&#xd;</xsl:text>

</xsl:template>

</xsl:stylesheet>