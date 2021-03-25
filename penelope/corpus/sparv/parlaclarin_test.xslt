<?xml version='1.0' encoding='UTF-8'?>
<xsl:stylesheet version="2.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:tei="http://www.tei-c.org/ns/1.0"
  xmlns="http://www.tei-c.org/ns/1.0"
  xmlns:xi="http://www.w3.org/2001/XInclude"
  exclude-result-prefixes="tei xi">

  <xsl:strip-space elements="*"/>
  <xsl:output indent="yes" method="text" encoding="utf-8" />

  <xsl:variable name="id" select="/tei:*/@xml:id"/>

  <xsl:template match="tei:u">

    <xsl:variable name="sid" select="@xml:id"/>
    <xsl:value-of select="$sid"></xsl:value-of>

    <xsl:choose>

        <xsl:when test="not(@xml:id) or $sid=''">
            <xsl:apply-templates/>

        </xsl:when>

        <xsl:otherwise>

            <xsl:value-of select="@who"></xsl:value-of>
            <xsl:text>&#9;</xsl:text>
            <xsl:value-of select="@xml:id"></xsl:value-of>
            <xsl:text>&#9;</xsl:text>
            <xsl:value-of select="@prev"></xsl:value-of>
            <xsl:text>&#9;</xsl:text>
            <xsl:value-of select="@next"></xsl:value-of>
            <xsl:text>&#9;</xsl:text>
            <xsl:apply-templates/>

        </xsl:otherwise>

    </xsl:choose>

  </xsl:template>

  <xsl:template match="tei:seg">
    <xsl:text>#</xsl:text>
    <xsl:value-of select="text()"></xsl:value-of>
    <xsl:text>#</xsl:text>
  </xsl:template>

  <xsl:template match="tei:note">
  </xsl:template>

  <xsl:template match="tei:front">
  </xsl:template>

  <xsl:template match="tei:teiHeader">
  </xsl:template>

</xsl:stylesheet>
