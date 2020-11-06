import zipfile

from penelope.utility import create_iterator
from tqdm.auto import tqdm

from .readers import TextTransformer


def preprocess_text_corpus(source_filename: str, target_filename: str, filename_pattern: str = '*.txt', _tqdm=tqdm):
    """Creates a preprocessed version of an archive

    Parameters
    ----------
    source_filename : str
        [description]
    target_filename : str
        [description]
    filename_pattern : str, optional
        [description], by default '*.txt'
    _tqdm : [type], optional
        [description], by default tqdm.tqdm
    """

    transformer = TextTransformer().fix_hyphenation().fix_unicode().fix_whitespaces().fix_ftfy()

    source = create_iterator(source_filename, filename_pattern=filename_pattern)

    if _tqdm is not None:
        source = _tqdm(source, desc='Preparing text corpus')

    with zipfile.ZipFile(target_filename, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, text in source:
            text = transformer.transform(text)
            zf.writestr(filename, text)
