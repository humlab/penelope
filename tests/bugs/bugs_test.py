import penelope.corpus.dtm as dtm
import penelope.notebook.interface as interface
import pytest
from penelope.corpus import ExtractTaggedTokensOpts, TextReaderOpts, TokensTransformOpts, VectorizeOpts
from penelope.pipeline import CorpusConfig
from penelope.pipeline.pipelines import wildcard_to_DTM_pipeline
from penelope.utility import PropertyValueMaskingOpts


@pytest.mark.skip(reason="Debug test")
def test_inidun_word_trends_bug():

    # phrases = parse_phrases(phrase_file, phrase)
    corpus_config: CorpusConfig = CorpusConfig.load("/data/inidun/SSI.yml")
    corpus_config.pipeline_payload.folders('/data/inidun')

    args: interface.ComputeOpts = interface.ComputeOpts(
        corpus_type=1,
        corpus_filename='/data/inidun/legal_instrument_corpus.zip',
        target_folder='./tests/output/MARS',
        corpus_tag='MARS',
        tokens_transform_opts=TokensTransformOpts(
            only_alphabetic=False,
            only_any_alphanumeric=False,
            to_lower=True,
            to_upper=False,
            min_len=1,
            max_len=None,
            remove_accents=False,
            remove_stopwords=False,
            stopwords=None,
            extra_stopwords=['örn'],
            language='english',
            keep_numerals=True,
            keep_symbols=True,
        ),
        text_reader_opts=TextReaderOpts(
            filename_pattern='*.txt',
            filename_filter=None,
            filename_fields=['unesco_id:_:2', 'year:_:3', 'city:\\w+\\_\\d+\\_\\d+\\_\\d+\\_(.*)\\.txt'],
            index_field=None,
            as_binary=False,
            sep='\t',
            quoting=3,
        ),
        extract_tagged_tokens_opts=ExtractTaggedTokensOpts(
            lemmatize=True,
            target_override=None,
            pos_includes='|NOUN|PROPN|DET|PRON|VERB|',
            pos_excludes='|PUNCT|EOL|SPACE|',
            pos_paddings='|ADJ|ADV|INTJ|PART|CONJ|CCONJ|SCONJ|NUM|AUX|SYM|X|ADP|',
            pos_replace_marker='*',
            passthrough_tokens=[],
            append_pos=False,
            phrases=None,
            to_lowercase=True,
        ),
        tagged_tokens_filter_opts=PropertyValueMaskingOpts(),
        vectorize_opts=VectorizeOpts(
            already_tokenized=True, lowercase=False, stop_words=None, max_df=1.0, min_df=1, verbose=False
        ),
        count_threshold=10,
        create_subfolder=True,
        persist=True,
        force=False,
        context_opts=None,
        partition_keys=None,
    )

    # workflows.document_term_matrix.compute(
    #     args=args,
    #     corpus_config=corpus_config,
    # )

    corpus: dtm.VectorizedCorpus = (
        corpus_config.get_pipeline(
            "tagged_frame_pipeline",
            corpus_filename=args.corpus_filename,
        )
        + wildcard_to_DTM_pipeline(
            tokens_transform_opts=args.tokens_transform_opts,
            extract_tagged_tokens_opts=args.extract_tagged_tokens_opts,
            tagged_tokens_filter_opts=args.tagged_tokens_filter_opts,
            vectorize_opts=args.vectorize_opts,
        )
    ).value()

    assert corpus is not None

    # if (args.count_threshold or 1) > 1:
    #     corpus = corpus.slice_by_n_count(args.count_threshold)

    # if args.persist:
    #     store_corpus_bundle(corpus, args)
