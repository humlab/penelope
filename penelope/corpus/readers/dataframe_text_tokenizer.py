import types

from nltk.tokenize import word_tokenize

from penelope.corpus.text_transformer import TRANSFORMS, TextTransformer


class DataFrameTextTokenizer:
    """Text iterator that returns row-wise text documents from a Pandas DataFrame
    """

    def __init__(self, df, **column_filters):
        """
        Parameters
        ----------
        df : DataFrame
            Data frame having one document per row. Text must be in column 'txt' and filename/id in `filename`
        """
        assert 'txt' in df.columns

        self.df = df

        if not 'filename' in self.df.columns:
            self.df['filename'] = self.df.index.astype(str)

        for column, value in column_filters.items():
            assert column in self.df.columns, column + ' is missing'
            if isinstance(value, tuple):
                assert len(value) == 2
                self.df = self.df[self.df[column].between(*value)]
            elif isinstance(value, list):
                self.df = self.df[self.df[column].isin(value)]
            else:
                self.df = self.df[self.df[column] == value]

        if len(self.df[self.df.txt.isna()]) > 0:
            print('Warn: {} n/a rows encountered'.format(len(self.df[self.df.txt.isna()])))
            self.df = self.df.dropna()

        self.text_transformer = TextTransformer(transforms=[])\
            .add(TRANSFORMS.fix_unicode)\
            .add(TRANSFORMS.fix_whitespaces)\
            .add(TRANSFORMS.fix_hyphenation)

        self.iterator = None
        self.metadata = self._compile_metadata()
        self.metadict = {x.filename: x for x in (self.metadata or [])}
        self.filenames = [x.filename for x in self.metadata]
        self.tokenize = word_tokenize

    def _create_iterator(self):
        return (self._process(row['filename'], row['txt']) for _, row in self.df.iterrows())

    def _process(self, filename: str, text: str):
        """Process the text and returns tokenized text
        """
        #text = self.preprocess(text)

        text = self.text_transformer.transform(text)

        tokens = self.tokenize(text, )

        return filename, tokens

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

    def _compile_metadata(self):
        """Returns document metadata as a list of dicts

        Returns
        -------
        List[types.SimpleNamespace]
            File meta data extracted from dataframe
        """
        assert 'filename' in self.df.columns

        df_m = self.df[[x for x in list(self.df.columns) if x != 'txt']]
        metadata = [types.SimpleNamespace(**meta) for meta in df_m.to_dict(orient='records')]

        return metadata
