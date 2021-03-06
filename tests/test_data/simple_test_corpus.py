import pandas as pd
from penelope.corpus import ReiterableTerms, metadata_to_document_index
from penelope.corpus.readers import TextReaderOpts
from penelope.utility import extract_filenames_metadata


class SimpleTestCorpus:
    def __init__(self, filename: str, reader_opts: TextReaderOpts):

        filename_fields = reader_opts.filename_fields
        index_field = reader_opts.index_field

        with open(filename, 'r') as f:
            lines = f.readlines()

        self.corpus_data = [
            dict(
                filename=data[0], title=data[1], text=data[2], tokens=[x.lower() for x in data[2].split() if len(x) > 0]
            )
            for data in [line.split(' # ') for line in lines]
        ]
        self.filenames = [x['filename'] for x in self.corpus_data]
        self.iterator = None

        metadata = extract_filenames_metadata(filenames=self.filenames, filename_fields=filename_fields)
        self.document_index: pd.DataFrame = metadata_to_document_index(metadata, document_id_field=index_field)
        self.document_index['title'] = [x['title'] for x in self.corpus_data]

    @property
    def terms(self):
        return ReiterableTerms(self)

    def _create_iterator(self):
        return ((x['filename'], x['tokens']) for x in self.corpus_data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iterator is None:
            self.iterator = self._create_iterator()
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = None
            raise
