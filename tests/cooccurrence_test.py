from penelope.cooccurrence import to_coocurrence_matrix, to_dataframe
from penelope.corpus import CorpusVectorizer, TokenizedCorpus
from penelope.corpus.readers import InMemoryReader

# from penelope.corpus.windowed_corpus import (concept_windows, corpus_concept_windows)

SPARV_ZIPPED_CSV_EXPORT_FILENAME = './tests/test_data/tranströmer_corpus_export.csv.zip'

# http://www.nltk.org/howto/collocations.html
# PMI


def test_cooccurrence_of_corpus_succeeds():

    expected_documents = [('tran_2019_01_test.txt', ['halvmörker', 'överblick', 'ljuslåga', 'människa']),
                          ('tran_2019_02_test.txt', ['strålkastarsken', 'människa', 'anletsdrag', 'mysterium']),
                          (
                              'tran_2019_03_test.txt', [
                                  'skäggstubb', 'sammanskruvade', 'grundsten', 'upplysning', 'tradition', 'anteckna',
                                  'invånare', 'grundsten', 'gångstig', 'kommunikationsnät', 'kraftledningsstolpen',
                                  'skalbagge', 'flygvingarna', 'hopvecklade', 'fallskärm'
                              ]
                          ), ('tran_2020_01_test.txt', ['kretsande', 'stillhet', 'fladdermus']),
                          ('tran_2020_02_test.txt', ['barrskogsbränningen', 'ögonblick'])]

    reader = InMemoryReader(expected_documents, filename_fields="year:_:1")
    corpus = TokenizedCorpus(reader=reader)

    assert expected_documents == [x for x in corpus]
    term_term_matrix = CorpusVectorizer()\
        .fit_transform(corpus, vocabulary=corpus.token2id)\
            .cooccurrence_matrix()

    assert term_term_matrix is not None
    assert term_term_matrix.shape == (26, 26)
    assert term_term_matrix.sum() == 120

    df_coo = to_dataframe(
        term_term_matrix=term_term_matrix, id2token=corpus.id2token, documents=corpus.documents, min_count=1
    )

    assert df_coo is not None

    #to_coocurrence_matrix
