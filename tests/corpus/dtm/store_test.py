import os
import shutil
import uuid
from io import StringIO
from os.path import join as jj

import numpy as np
import pytest

from penelope.corpus import (
    CorpusVectorizer,
    TokenizedCorpus,
    TokensTransformOpts,
    VectorizedCorpus,
    load_metadata,
    store_metadata,
)
from penelope.corpus.document_index import load_document_index
from tests.utils import OUTPUT_FOLDER, create_tokens_reader

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# pylint: disable=redefined-outer-name


@pytest.fixture
def text_corpus() -> TokenizedCorpus:
    filename_fields = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
    reader = create_tokens_reader(filename_fields=filename_fields, text_transforms="dehyphen,normalize-whitespace")
    transform_opts = TokensTransformOpts(
        transforms={
            'only-any-alphanumeric': True,
            'to-lower': True,
            'min-chars': 2,
            'remove-numerals': True,
        }
    )
    corpus = TokenizedCorpus(reader, transform_opts=transform_opts)
    return corpus


@pytest.fixture
def vectorized_corpus(text_corpus: TokenizedCorpus) -> VectorizedCorpus:
    corpus: VectorizedCorpus = CorpusVectorizer().fit_transform(text_corpus, already_tokenized=True)
    return corpus


@pytest.mark.parametrize('mode', ['bundle', 'files'])
def test_load_stored_metadata_simple(mode: str):
    tag: str = f'{uuid.uuid1()}'
    folder: str = jj(OUTPUT_FOLDER, tag)

    os.makedirs(folder, exist_ok=True)

    document_index_str: str = (
        ";filename;year;document_name;document_id\n"
        "a;a.txt;2019;a;0\n"
        "b;b.txt;2019;b;1\n"
        "c;c.txt;2019;c;2\n"
        "d;d.txt;2020;d;3\n"
        "e;e.txt;2020;e;4\n"
    )
    token2id: dict = dict(x=0, y=1, z=2)
    overridden_term_frequency = np.arange(3)

    metadata = {
        'document_index': load_document_index(filename=StringIO(document_index_str), sep=';'),
        'token2id': token2id,
        'overridden_term_frequency': overridden_term_frequency,
    }

    store_metadata(tag=tag, folder=folder, mode=mode, **metadata)

    metadata_loaded = load_metadata(tag=tag, folder=folder)
    assert metadata_loaded['document_index'].to_csv(sep=';') == document_index_str
    assert metadata_loaded['token2id'] == token2id
    assert (metadata_loaded['overridden_term_frequency'] == overridden_term_frequency).all()

    shutil.rmtree(folder)


@pytest.mark.parametrize('mode', ['bundle', 'files'])
def test_load_stored_metadata(mode: str, vectorized_corpus: VectorizedCorpus):
    tag: str = f'{uuid.uuid1()}'
    folder: str = jj(OUTPUT_FOLDER, tag)

    os.makedirs(folder, exist_ok=True)

    store_metadata(tag=tag, folder=folder, mode=mode, **vectorized_corpus.metadata)

    metadata = load_metadata(tag=tag, folder=folder)

    assert metadata is not None
    assert metadata['token2id'] == vectorized_corpus.token2id
    assert (metadata['document_index'] == vectorized_corpus.document_index).all().all()
    assert metadata['overridden_term_frequency'] == vectorized_corpus.overridden_term_frequency

    shutil.rmtree(folder)


@pytest.mark.parametrize('mode', ['bundle', 'files'])
def test_load_dumped_corpus(mode: str, vectorized_corpus: VectorizedCorpus):
    tag: str = f'{str(uuid.uuid1())[:6]}'
    folder: str = jj(OUTPUT_FOLDER, tag)

    os.makedirs(folder, exist_ok=True)

    vectorized_corpus.dump(tag=tag, folder=folder, compressed=True, mode=mode)

    assert VectorizedCorpus.dump_exists(tag=tag, folder=folder)
    assert VectorizedCorpus.find_tags(folder) == [tag]
    assert VectorizedCorpus.is_dump(jj(folder, f'{tag}_vector_data.npz'))

    for args in [{'tag': tag, 'folder': folder}, {'filename': jj(folder, f'{tag}_vector_data.npz')}]:
        loaded_corpus: VectorizedCorpus = VectorizedCorpus.load(**args)
        assert (vectorized_corpus.term_frequency == loaded_corpus.term_frequency).all()
        assert vectorized_corpus.document_index.to_dict() == loaded_corpus.document_index.to_dict()
        assert vectorized_corpus.token2id == loaded_corpus.token2id

    loaded_corpus: VectorizedCorpus = VectorizedCorpus.load(filename=jj(folder, f'{tag}_vector_data.npz'))

    loaded_options: dict = VectorizedCorpus.load_options(tag=tag, folder=folder)
    assert loaded_options == dict()

    VectorizedCorpus.dump_options(tag=tag, folder=folder, options=dict(apa=1))
    loaded_options: dict = VectorizedCorpus.load_options(tag=tag, folder=folder)
    assert loaded_options == dict(apa=1)

    VectorizedCorpus.remove(tag=tag, folder=folder)
    assert not VectorizedCorpus.dump_exists(tag=tag, folder=folder)
    assert not VectorizedCorpus.find_tags(folder)

    shutil.rmtree(folder)

    assert VectorizedCorpus.split(jj(folder, f'{tag}_vector_data.npz')) == (folder, tag)
    assert VectorizedCorpus.split(jj(folder, f'{tag}_vector_data.npy')) == (folder, tag)
    assert VectorizedCorpus.split(jj(folder, f"{tag}_vectorizer_data.pickle")) == (folder, tag)
