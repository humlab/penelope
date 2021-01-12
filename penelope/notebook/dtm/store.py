import os

import penelope.corpus.dtm as dtm

from ..interface import ComputeOpts


def store_corpus_bundle(corpus: dtm.VectorizedCorpus, args: ComputeOpts):

    if dtm.VectorizedCorpus.dump_exists(tag=args.corpus_tag, folder=args.target_folder):
        dtm.VectorizedCorpus.remove(tag=args.corpus_tag, folder=args.target_folder)

    target_folder = args.target_folder

    if args.create_subfolder:
        if os.path.split(target_folder)[1] != args.corpus_tag:
            target_folder = os.path.join(target_folder, args.corpus_tag)
        os.makedirs(target_folder, exist_ok=True)

    corpus.dump(tag=args.corpus_tag, folder=target_folder)

    dtm.VectorizedCorpus.dump_options(
        tag=args.corpus_tag,
        folder=target_folder,
        options={
            'input_filename': args.corpus_filename,
            'output_folder': target_folder,
            'output_tag': args.corpus_tag,
            'count_threshold': args.count_threshold,
            'reader_opts': args.text_reader_opts.props,
            'tokens_transform_opts': args.tokens_transform_opts.props,
            'extract_tokens_opts': args.extract_tagged_tokens_opts.props
            if args.extract_tagged_tokens_opts is not None
            else {},
            'vectorize_opt': args.vectorize_opts.props,
        },
    )
