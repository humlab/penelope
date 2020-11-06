import json
import os
from typing import Any

from penelope.corpus import CorpusVectorizer, TextTransformOpts, TokenizedCorpus, TokensTransformOpts, VectorizedCorpus
from penelope.corpus.readers import TextTokenizer
from penelope.utility import getLogger, replace_extension

from .utils import WorkflowException

logger = getLogger("penelope")

# pylint: disable=too-many-arguments


def execute_workflow(
    input_filename: str,
    output_folder: str,
    output_tag: str,
    *,
    filename_field: Any = None,
    filename_pattern: str = '*.*',
    count_threshold: int = None,
    tokens_transform_opts: TokensTransformOpts = None,
):

    if not os.path.isfile(input_filename):
        logger.error('no such file: {}'.format(input_filename))
        return

    if len(filename_field or []) == 0:
        raise WorkflowException("please specify at least one filename field (e.g. --filename-field='year:_:1')")

    logger.info('Creating new corpus...')

    if VectorizedCorpus.dump_exists(tag=output_tag, folder=output_folder):
        logger.info('removing existing result files...')
        os.remove(os.path.join(output_folder, '{}_vector_data.npy'.format(output_tag)))
        os.remove(os.path.join(output_folder, '{}_vectorizer_data.pickle'.format(output_tag)))
        os.remove(os.path.join(output_folder, '{}_vectorizer_data.json'.format(output_tag)))

    tokenizer_opts = dict(
        filename_pattern=filename_pattern,
        filename_fields=filename_field,
        as_binary=False,
        tokenize=None,
    )

    corpus = TokenizedCorpus(
        TextTokenizer(
            source=input_filename,
            **tokenizer_opts,
            text_transform_opts=TextTransformOpts(fix_whitespaces=True, fix_hyphenation=True),
        ),
        tokens_transform_opts=tokens_transform_opts,
    )

    logger.info('Creating document-term matrix...')
    vectorizer = CorpusVectorizer()
    v_corpus = vectorizer.fit_transform(corpus)

    logger.info('Saving data matrix...')
    v_corpus.dump(tag=output_tag, folder=output_folder)

    logger.info('Saving parameters JSON...')
    json_filename = os.path.join(output_folder, f"{output_tag}_vectorizer_data.json")
    with open(replace_extension(json_filename, 'json'), 'w') as json_file:
        store_options = {
            'input': input_filename,
            'output': output_folder,
            'tag': output_tag,
            'n_count_threshold': count_threshold,
            'tokenizer_opts': tokenizer_opts,
            'tokens_transform_opts': tokens_transform_opts.props,
            'annotation_opts': {},
        }
        json.dump(store_options, json_file, indent=4)
