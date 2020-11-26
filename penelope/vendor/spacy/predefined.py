import pandas as pd
from penelope.corpus import VectorizedCorpus, VectorizeOpts
from penelope.corpus.readers import ExtractTokensOpts2, TextReaderOpts, TextSource, TextTransformOpts
from spacy.language import Language

from . import interfaces
from . import pipeline as corpus_pipeline


def extract_text_to_vectorized_corpus(
    source: TextSource,
    nlp: Language,
    *,
    reader_opts: TextReaderOpts,
    transform_opts: TextTransformOpts,
    extract_tokens_opts: ExtractTokensOpts2,
    vectorize_opts: VectorizeOpts,
    document_index: pd.DataFrame = None,
) -> VectorizedCorpus:
    payload = interfaces.PipelinePayload(source=source, document_index=document_index)
    pipeline = (
        corpus_pipeline.SpacyPipeline(payload=payload)
        .load(reader_opts=reader_opts, transform_opts=transform_opts)
        .text_to_spacy(nlp=nlp)
        .spacy_to_dataframe(nlp=nlp, attributes=['text', 'lemma_', 'pos_'])
        .dataframe_to_tokens(extract_tokens_opts=extract_tokens_opts)
        .tokens_to_text()
        .to_dtm(vectorize_opts)
    )

    corpus = pipeline.resolve()

    return corpus
