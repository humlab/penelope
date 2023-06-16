from os.path import splitext

import pandas as pd

from penelope.common.render_text import DEFAULT_TEMPLATE, RenderService, TextRepository
from penelope.corpus import load_document_index

LINKS_REGISTRY = {'PDF': lambda _: 'PDF', 'MD': lambda _: 'MD'}


def test_text_repository():
    document_index: pd.DataFrame = load_document_index('tests/test_data/tranströmer_corpus.csv', sep="\t")
    repository: TextRepository = TextRepository(
        source='tests/test_data/tranströmer_corpus.zip',
        document_index=document_index,
    )

    assert repository is not None

    filenames: list[str] = repository.filenames

    assert len(filenames) == 5
    assert len(filenames) == len(document_index)

    filename: str = 'tran_2019_03_test.txt'
    expected_text: str = 'Det finns mitt i skogen en oväntad glänta'
    document_name: str = splitext(filename)[0]

    text: str = repository.get_text(filename)

    assert isinstance(text, str)
    assert text.startswith(expected_text)

    text: str = repository.get_text(document_name)

    assert isinstance(text, str)
    assert text.startswith(expected_text)

    data: dict = repository.get(document_name)

    assert isinstance(data, dict)
    assert data.get('document_name') == document_name
    assert data.get('text', "").startswith(expected_text)

    document_name: str = 'DOESNOTEXIST.txt'
    text: str = repository.get_text(document_name)

    assert isinstance(text, str)
    assert text.startswith('document not found')

    data: dict = repository.get(document_name)

    assert isinstance(data, dict)
    assert data.get('text', '').startswith('document not found')


def test_text_repository_trender_document():
    template = DEFAULT_TEMPLATE
    renderer: RenderService = RenderService(template, links_registry=LINKS_REGISTRY)

    document_info: dict = {
        'filename': 'tran_2019_03_test.csv',
        'year': 2019,
        'number': 3,
        'document_id': 2,
        'document_name': 'tran_2019_03_test',
        'text': 'Det finns mitt i skogen en oväntad glänta',
    }
    text: str = renderer.render(document_info=document_info, kind='text')
    assert text == document_info['text']

    html: str = renderer.render(document_info=document_info, kind='html')

    assert document_info['text'] in html
    assert '<b>Document:</b> tran_2019_03_test (2) <br/>' in html
