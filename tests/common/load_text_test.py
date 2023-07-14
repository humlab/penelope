from penelope.corpus import dehyphen
from penelope.corpus.render import LemmaCorpusLoader, TaggedCorpusLoader, ZipLoader, ZippedTextCorpusLoader

# pylint: disable=protected-access


def test_remove_hyphens():
    text: str = """The choreo-
 graphy which makes one's
head spin in its com¬
plex pat-

terns has
been evolved through hun-dreds of years
of tra-
    dition and training."""

    result = dehyphen(text)

    assert (
        result
        == """The choreography
which makes one's
head spin in its complex
patterns
has
been evolved through hun-dreds of years
of tradition
and training."""
    )


TEST_DOCUMENT_INFO: dict = {
    'filename': 'tran_2019_03_test.csv',
    'year': 2019,
    'number': 3,
    'document_id': 2,
    'document_name': 'tran_2019_03_test',
    'text': 'Det finns mitt i skogen en oväntad glänta',
}


def test_load_tagged_article_document_text():
    document_name: str = 'tran_2019_03_test'

    source: str = 'tests/test_data/tranströmer/tranströmer_corpus.zip'

    text = ZipLoader(source).load(f'{document_name}.txt')
    assert (text or '').startswith('Det finns mitt i skogen en oväntad glänta')

    text = ZippedTextCorpusLoader(source).load(document_name)
    assert (text or '').startswith('Det finns mitt i skogen en oväntad glänta')

    text = TaggedCorpusLoader(source).load(document_name)
    assert (text or '').startswith("# text Det finns mitt i skogen en oväntad glänta")

    text = LemmaCorpusLoader(source).load(document_name)
    assert (text or '').startswith('|den|en| |finna|finnas| |mitt|mitt i| |mitt i:3| |skog| |en|')
