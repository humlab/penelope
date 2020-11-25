from penelope.corpus.readers.interfaces import TextReaderOpts
from tests.test_data.simple_test_corpus import SimpleTestCorpus


class TranströmerCorpus(SimpleTestCorpus):
    def __init__(self):
        # tran_2019_02_test.txt
        meta_fields = ["year:_:1", "year_serial_id:_:2"]
        super().__init__('./tests/test_data/tranströmer.txt', TextReaderOpts(filename_fields=meta_fields))
