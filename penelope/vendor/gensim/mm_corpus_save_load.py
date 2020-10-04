import logging
import os

import gensim
import pandas as pd

from .ext_mm_corpus import ExtMmCorpus

logger = logging.getLogger(__name__)

jj = os.path.join


def _mm_filename(source_folder, lang):
    return jj(source_folder, f'corpus_{lang}.mm')


def _dict_filename(source_folder, lang):
    return jj(source_folder, f'corpus_{lang}.dict.gz')


def _documents_filename(source_folder, lang):
    return jj(source_folder, f'corpus_{lang}_documents.csv')


def store_as_mm_corpus(source_folder, lang, corpus):

    gensim.corpora.MmCorpus.serialize(_mm_filename(source_folder, lang), corpus, id2word=corpus.dictionary.id2token)

    corpus.dictionary.save(_dict_filename(source_folder, lang))
    corpus.document_names.to_csv(_documents_filename(source_folder, lang), sep='\t')


def load_mm_corpus(source_folder, lang, normalize_by_D=False):

    corpus_type = ExtMmCorpus if normalize_by_D else gensim.corpora.MmCorpus

    corpus = corpus_type(_mm_filename(source_folder, lang))

    corpus.dictionary = gensim.corpora.Dictionary.load(_dict_filename(source_folder, lang))
    corpus.document_names = pd.read_csv(_documents_filename(source_folder, lang), sep='\t').set_index('document_id')

    return corpus


def exists(source_folder, lang):
    return (
        os.path.isfile(_mm_filename(source_folder, lang))
        and os.path.isfile(_dict_filename(source_folder, lang))
        and os.path.isfile(_documents_filename(source_folder, lang))
    )
