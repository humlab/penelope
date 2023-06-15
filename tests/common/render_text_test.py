from os.path import splitext

import pandas as pd

from penelope.common.render_text import DEFAULT_TEMPLATE, RenderService, TextRepository
from penelope.corpus import load_document_index

LINKS_REGISTRY = {'PDF': lambda _: 'PDF', 'MD': lambda _: 'MD'}


def test_text_repository():
    document_index: pd.DataFrame = load_document_index('tests/test_data/tranströmer_corpus.csv', sep=";")
    repository: TextRepository = TextRepository(
        source='tests/test_data/tranströmer_corpus.zip',
        document_index=document_index,
    )

    assert repository is not None

    filenames: list[str] = repository.filenames

    assert len(filenames) == 5
    assert len(filenames) == len(document_index)

    filename: str = 'CONSTITUTION_0201_015244_1945_london.txt'
    expected_text: str = 'Constitution of the United Nations Educational, Sc'
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
    renderer: RenderService = RenderService(template, links_registry=LINKS_REGISTRY.items())

    document_info: dict = {
        'document_id': 1,
        'document_name': 'CONSTITUTION_0201_015244_1945_london',
        'text': 'Constitution of the United Nations Educational, Sc',
        'num_tokens': 100,
        'num_types': 50,
    }
    text: str = renderer.render(document_info=document_info, kind='text')
    assert text == document_info['text']

    html: str = renderer.render(document_info=document_info, kind='html')

    assert document_info['text'] in html
    assert '<b>Document:</b> CONSTITUTION_0201_015244_1945_london (1) <br/>' in html
