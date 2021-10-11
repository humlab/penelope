import uuid

from penelope.corpus import ExtractTaggedTokensOpts, TextReaderOpts, TokensTransformOpts
from penelope.pipeline import CorpusConfig, CorpusPipeline, DocumentPayload
from penelope.utility import PropertyValueMaskingOpts

DEFAULT_ENGINE_ARGS = {
    'n_topics': 4,
    'passes': 1,
    'random_seed': 42,
    'alpha': 'auto',
    'workers': 1,
    'max_iter': 100,
    'prefix': './tests/output/',
}


def ssi_corpus_config() -> CorpusConfig:
    config: CorpusConfig = CorpusConfig.load('./tests/test_data/SSI.yml')
    config.pipeline_payload.source = './tests/test_data/legal_instrument_five_docs_test.zip'
    config.pipeline_payload.document_index_source = './tests/test_data/legal_instrument_five_docs_test.csv'
    return config


def ssi_topic_model_payload(config: CorpusConfig, en_nlp) -> DocumentPayload:
    target_name: str = f'{uuid.uuid1()}'
    transform_opts: TokensTransformOpts = TokensTransformOpts()
    reader_opts: TextReaderOpts = TextReaderOpts(filename_pattern="*.txt", filename_fields="year:_:1")
    extract_opts: ExtractTaggedTokensOpts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes='|VERB|NOUN|', pos_paddings='|ADJ|', **config.pipeline_payload.tagged_columns_names
    )
    transform_opts = None
    filter_opts: PropertyValueMaskingOpts = PropertyValueMaskingOpts(is_punct=False)
    payload: DocumentPayload = (
        CorpusPipeline(config=config)
        .load_text(reader_opts=reader_opts, transform_opts=TokensTransformOpts())
        .set_spacy_model(en_nlp)
        .text_to_spacy()
        .spacy_to_tagged_frame(attributes=['text', 'lemma_', 'pos_'])
        .tagged_frame_to_tokens(extract_opts=extract_opts, filter_opts=filter_opts, transform_opts=transform_opts)
        .to_topic_model(
            corpus_filename=None,
            target_folder="./tests/output",
            target_name=target_name,
            engine="gensim_lda-multicore",
            engine_args=DEFAULT_ENGINE_ARGS,
            store_corpus=True,
            store_compressed=True,
        )
    ).single()
    return payload
