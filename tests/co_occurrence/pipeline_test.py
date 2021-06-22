import os
import uuid
from typing import Iterable

import pytest
import scipy.sparse as sp
from penelope.co_occurrence import Bundle, ContextOpts, VectorizeType
from penelope.corpus import ClosedVocabularyError, TokenizedCorpus, VectorizedCorpus
from penelope.pipeline import CorpusConfig, CorpusPipeline, DocumentPayload, PipelinePayload
from penelope.pipeline.co_occurrence.tasks import CoOccurrencePayload
from penelope.pipeline.config import CorpusType

from ..fixtures import SIMPLE_CORPUS_ABCDE_5DOCS, very_simple_corpus

jj = os.path.join

CONTEXT_OPTS: ContextOpts = ContextOpts(context_width=2, concept={}, ignore_concept=False, ignore_padding=False)


@pytest.mark.skip(reason="ingest is now prohibited if vocabulary is closed")
def test_pipeline_to_co_occurrence_ingest_prohobited_if_vocabulary_exists():

    tokenized_corpus: TokenizedCorpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)
    config: CorpusConfig = CorpusConfig.tokenized_corpus_config()

    ingest_tokens = True
    with pytest.raises(ClosedVocabularyError):
        _: Bundle = (
            CorpusPipeline(config=config)
            .load_corpus(tokenized_corpus)
            .vocabulary(lemmatize=False)
            .to_document_co_occurrence(context_opts=CONTEXT_OPTS, ingest_tokens=ingest_tokens)
            .to_corpus_co_occurrence(context_opts=CONTEXT_OPTS, global_threshold_count=1)
            .single()
            .content
        )


def test_pipeline_to_co_occurrence_can_create_new_vocabulary():

    tokenized_corpus: TokenizedCorpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)
    config: CorpusConfig = CorpusConfig.tokenized_corpus_config()

    ingest_tokens = False
    bundle: Bundle = (
        CorpusPipeline(config=config)
        .load_corpus(tokenized_corpus)
        .vocabulary(lemmatize=False)
        .to_document_co_occurrence(context_opts=CONTEXT_OPTS, ingest_tokens=ingest_tokens)
        .to_corpus_co_occurrence(context_opts=CONTEXT_OPTS, global_threshold_count=1)
        .single()
        .content
    )
    assert isinstance(bundle, Bundle)
    corpus: VectorizedCorpus = bundle.corpus
    assert isinstance(corpus, VectorizedCorpus)


def simple_co_occurrence(windows: Iterable[Iterable[str]]):
    counter: dict = dict()
    for window in windows:
        for i in range(0, len(window) - 1):
            for j in range(i + 1, len(window)):
                t1, t2 = window[i], window[j]
                if t1 == t2:
                    continue
                pair = (t1, t2) if t1 < t2 else (t2, t1)
                counter[pair] = counter.get(pair, 0) + 1
    return counter


def test_pipeline_to_co_occurrence_can_create_co_occurrence_bundle():
    context_opts: ContextOpts = ContextOpts(context_width=2, concept={}, ignore_concept=False, ignore_padding=False)
    tokenized_corpus: TokenizedCorpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)
    config: CorpusConfig = CorpusConfig(
        corpus_name=uuid.uuid1(),
        corpus_type=CorpusType.Tokenized,
        corpus_pattern=None,
        checkpoint_opts=None,
        text_reader_opts=None,
        filter_opts=None,
        pipelines=None,
        pipeline_payload=PipelinePayload(),
        language="swedish",
    )

    """Expected windows generated for corpus"""
    # print({ k: [x for x in generate_windows(tokens=tokens, context_opts=context_opts)] for k, tokens in expected_tokens.items() })
    document_windows = {
        'tran_2019_01_test.txt': [
            ['*', '*', 'a', 'b', 'c'],
            ['*', 'a', 'b', 'c', 'c'],
            ['a', 'b', 'c', 'c', '*'],
            ['b', 'c', 'c', '*', '*'],
        ],
        'tran_2019_02_test.txt': [
            ['*', '*', 'a', 'a', 'b'],
            ['*', 'a', 'a', 'b', 'd'],
            ['a', 'a', 'b', 'd', '*'],
            ['a', 'b', 'd', '*', '*'],
        ],
        'tran_2019_03_test.txt': [
            ['*', '*', 'a', 'e', 'e'],
            ['*', 'a', 'e', 'e', 'b'],
            ['a', 'e', 'e', 'b', '*'],
            ['e', 'e', 'b', '*', '*'],
        ],
        'tran_2020_01_test.txt': [
            ['*', '*', 'c', 'c', 'd'],
            ['*', 'c', 'c', 'd', 'a'],
            ['c', 'c', 'd', 'a', '*'],
            ['c', 'd', 'a', '*', '*'],
        ],
        'tran_2020_02_test.txt': [
            ['*', '*', 'a', 'b', 'b'],
            ['*', 'a', 'b', 'b', 'e'],
            ['a', 'b', 'b', 'e', '*'],
            ['b', 'b', 'e', '*', '*'],
        ],
    }

    """Expected co-occurrences from windows above"""
    expected_TTMs = {filename: simple_co_occurrence(document_windows[filename]) for filename in document_windows}

    def verify_tokens_payload(
        p: CorpusPipeline, payload: DocumentPayload, *_  # pylint: disable=unused-argument
    ) -> bool:
        # expected_tokens: dict = { k: v for k, v in SIMPLE_CORPUS_ABCDE_5DOCS}

        expected_tokens: dict = {
            'tran_2019_01_test.txt': ['a', 'b', 'c', 'c'],
            'tran_2019_02_test.txt': ['a', 'a', 'b', 'd'],
            'tran_2019_03_test.txt': ['a', 'e', 'e', 'b'],
            'tran_2020_01_test.txt': ['c', 'c', 'd', 'a'],
            'tran_2020_02_test.txt': ['a', 'b', 'b', 'e'],
        }

        return payload.content == expected_tokens.get(payload.filename)

    def verify_expected_vocabulary(p: CorpusPipeline, *_) -> bool:
        return list(p.payload.token2id.keys()) == ['*', '__low-tf__', 'a', 'b', 'c', 'd', 'e']

    def verify_co_occurrence_document_TTM_payload(
        p: CorpusPipeline, payload: DocumentPayload, *_
    ) -> bool:  # pylint: disable=unused-argument

        fg = p.payload.token2id.id2token.get

        assert isinstance(payload.content, CoOccurrencePayload)

        TTM: sp.spmatrix = payload.content.vectorized_data.get(VectorizeType.Normal).term_term_matrix.tocoo()

        document_TTM_data = {(fg(TTM.row[i]), fg(TTM.col[i])): TTM.data[i] for i in range(0, len(TTM.data))}

        assert expected_TTMs[payload.filename] == document_TTM_data

        return True

    bundle: Bundle = (
        CorpusPipeline(config=config)
        .load_corpus(tokenized_corpus)
        .assert_on_payload(payload_test=verify_tokens_payload)
        .vocabulary(lemmatize=True)
        .assert_on_exit(exit_test=verify_expected_vocabulary)
        .to_document_co_occurrence(context_opts=context_opts, ingest_tokens=False)
        .assert_on_payload(payload_test=verify_co_occurrence_document_TTM_payload)
        .to_corpus_co_occurrence(context_opts=context_opts, global_threshold_count=1)
        .single()
        .content
    )

    for filename in expected_TTMs:
        document_id = int(bundle.document_index[bundle.document_index.filename == filename].document_id)
        for (i, j), ij in bundle.vocabs_mapping.items():
            pair = (bundle.token2id.id2token[i], bundle.token2id.id2token[j])
            if pair in expected_TTMs[filename]:
                assert bundle.corpus.data[document_id, ij] == expected_TTMs[filename][pair]
            else:
                assert bundle.corpus.data[document_id, ij] == 0
