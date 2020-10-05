#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This source code is based on ...
#
# Copyright (C) ....
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
r"""Wrapper for `The Stockholm NER tagger(Stagger) NER tagger <https://www.ling.su.se/english/nlp/tools/stagger>`_

Notes ==================>    NOT IMPLEMENTED!!!!
-----

Installation
------------

Examples
--------

"""
import datetime
import logging
import os

from gensim.utils import check_output

logger = logging.getLogger(__name__)


class StaggerWrapper:
    """Wrapper for The Stockholm NER tagger(Stagger) NER tagger
    <https://www.ling.su.se/english/nlp/tools/stagger>.
    """

    def __init__(self, stagger_jar_path=None, stagger_model_path=None):
        """
        Parameters
        ----------
        stagger_jar_path : str
            Path to the Stagger jar file.
        stagger_model_path : str
            Path to the Stagger model file.
        """

        self.word_topics = None
        self.wordtopics = None

        self.stagger_home = os.environ.get('STAGGER_HOME', None)
        self.stagger_jar_path = stagger_jar_path
        self.stagger_model_path = stagger_model_path

        if self.stagger_jar_path is None and self.stagger_home is not None:
            self.stagger_jar_path = os.path.join(self.stagger_home, "stagger.jar")

        if self.stagger_model_path is None and self.stagger_home is not None:
            self.stagger_model_path = os.path.join(self.stagger_home, 'swedish.bin')

        if not os.path.exists(self.stagger_jar_path or ""):
            raise FileNotFoundError("Stagger jar file not found or not specified")

        if not os.path.exists(self.stagger_model_path or ""):
            raise FileNotFoundError("Stagger model file not found or not specified")

        self.name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def build_command(self, file_pattern, memory_size):
        return ' '.join(
            [
                'java',
                '-Xmx{}'.format(memory_size),
                '-jar',
                self.stagger_jar_path,
                '-modelfile',
                self.stagger_model_path,
                '-lang',
                'sv',
                '-tag',
                file_pattern,
            ]
        )

    def stagit(self, corpus_files, memory_size='4G'):
        """Tag corpus.

        Parameters
        ----------
        corpus_path : file pattern that specifies corpus files

        """

        # java -Xmx4G -jar stagger.jar -modelfile models/swedish.bin -tag test.txt > test.conll

        if isinstance(corpus_files, list):
            files_or_pattern = ' '.join(corpus_files)
        else:
            files_or_pattern = corpus_files

        command = self.build_command(files_or_pattern, memory_size)

        check_output(args=command, shell=True)

        # self.word_topics = self.load_word_topics()
        # self.wordtopics = self.word_topics
