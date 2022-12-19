import zipfile

import pandas as pd

from penelope.utility import getLogger, path_add_sequence

logger = getLogger("")


# TODO: Check for duplicate store elsewhere, e.g. zip_util.store
def store_tokenized_corpus_as_archive(tokenized_docs, target_filename):
    """Stores a tokenized (string) corpus to a zip archive

    Parameters
    ----------
    tokenized_docs : [type]
        [description]
    corpus_source_filepath : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    file_stats = []
    process_count = 0

    # TODO: Enable store of all documents line-by-line in a single file
    with zipfile.ZipFile(target_filename, "w", compression=zipfile.ZIP_DEFLATED) as zf:

        for document_id, filename, chunk_index, tokens in tokenized_docs:

            text = ' '.join([t.replace(' ', '_') for t in tokens])
            store_name = path_add_sequence(filename, chunk_index, 4)

            zf.writestr(store_name, text, compress_type=zipfile.ZIP_DEFLATED)

            file_stats.append((document_id, filename, chunk_index, len(tokens)))

            if process_count % 100 == 0:
                logger.info('Stored {} files...'.format(process_count))

            process_count += 1

    df_summary = pd.DataFrame(file_stats, columns=['document_id', 'filename', 'chunk_index', 'n_tokens'])

    return df_summary
