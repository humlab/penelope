"""
Author: Stian Rødven Eide

This is in implementation of method described in "Anforanden: Annotated and Augmented Parliamentary Debates from Sweden", Stian Rødven Eide
Published in PARLACLARIN 2020. https://www.semanticscholar.org/paper/Anf%C3%B6randen%3A-Annotated-and-Augmented-Parliamentary-Eide/46baeb3f444a085540a1b57278de7ed4ea385b04

This source code is heavily influenced by the source code found at https://gitlab.com/Julipan/swedish-de-hyphenator/, released under GNU GPLv3.

License: https://gitlab.com/Julipan/swedish-de-hyphenator/-/blob/master/LICENSE

"""
import re
from dataclasses import dataclass, field
from enum import IntEnum
from os.path import isfile, join
from typing import Callable

from loguru import logger

from ..inject import ConfigValue
from .utility import load_dict, load_token_set, store_dict, store_token_set

PARAGRAPH_MARKER = '##PARAGRAPH##'

# pylint: disable=too-many-arguments
jj = join


class WhitelistReason(IntEnum):
    Undefined = 0
    HyphenatedCompound = 1
    Frequency = 2
    UnknownParts = 3


IGNORE_CONJUNCTION_WORDS: set[str] = {
    'än',
    'eller',
    'framför',
    'inklusive',
    'kontra',
    'liksom',
    'men',
    'o',
    'och',
    'og',
    'respektive',
    'samt',
    'såväl',
    'snart',
    'som',
    'till',
    'und',
    'utan',
}


def is_ignored_by_conjunction_word(dashed_word: str) -> bool:
    return any(dashed_word.endswith(f' {w}') for w in IGNORE_CONJUNCTION_WORDS)


def find_dashed_words(text: str) -> set[str]:
    dashed_words = [d for d in re.findall(r'\w+- \w+', text) if not is_ignored_by_conjunction_word(d)]
    return dashed_words


class ParagraphMergeStrategy(IntEnum):
    DoNotMerge = (0,)
    MergeIfWordsOnlySeparatedByTwoNewlines = (1,)
    MergeAll = 2


WORD_FREQUENCIES_NAME: str = 'word-frequencies.pkl'


@dataclass
class SwedishDehyphenator:
    """Dehyphens Swedish text"""

    data_folder: str = ConfigValue.create_field(key="dehyphen:folder", default="/data")
    word_frequencies: dict[str, int] | str = ConfigValue.create_field(
        key="dehyphen:tf_filename,tf_filename", default=None
    )
    paragraph_merge_strategy: ParagraphMergeStrategy = 0

    # Internal data
    whitelist: set[str] = field(default_factory=set)
    whitelist_log: dict[str, int] = field(default_factory=dict)
    unresolved: set[str] = field(default_factory=set)
    filename: str = field(init=False, default=WORD_FREQUENCIES_NAME)

    def __post_init__(self) -> "SwedishDehyphenator":
        self.data_folder = self.data_folder or "."

        if isinstance(self.word_frequencies, dict):
            self.filename = jj(self.data_folder, self.filename)
        else:
            if isinstance(self.word_frequencies, str):
                self.filename = self.word_frequencies
                self.word_frequencies = None

            self.filename: str = self.probe_word_frequencies_filename()

            self.word_frequencies = load_dict(self.filename)

        self.load_state()

    def probe_word_frequencies_filename(self) -> str:
        if isfile(self.filename):
            return self.filename

        if isfile(jj(self.data_folder, self.filename)):
            return jj(self.data_folder, self.filename)

        raise FileNotFoundError(f"expected TF file {self.filename} not found")

    @property
    def whitelist_filename(self) -> str:
        return jj(self.data_folder, 'dehyphen_whitelist.txt.gz')

    @property
    def whitelist_log_filename(self) -> str:
        return jj(self.data_folder, 'dehyphen_whitelist_log.pkl')

    @property
    def unresolved_filename(self) -> str:
        return jj(self.data_folder, 'dehyphen_unresolved.txt.gz')

    def is_whitelisted(self, word: str) -> bool:
        return word.lower() in self.whitelist

    def is_known_word(self, word: str) -> bool:
        return word in self.whitelist

    def add_to_whitelist(self, word: str, reason_code: WhitelistReason = WhitelistReason.Undefined):
        self.whitelist.add(word.lower())
        if word in self.unresolved:
            self.unresolved.remove(word)
        self.whitelist_log[word] = int(reason_code)
        return word

    @staticmethod
    def is_hyphenated_compound(dashed_word: str) -> bool:
        """Test if is compund"""
        if re.match(
            r'[A-ZÅÄÖ]+-[a-zåäö]+|' + r'[A-ZÅÄÖ][a-zåäö]+-[A-ZÅÄÖ][a-zåäö]+|' + r'\d+-\w+|' + r'icke-\w+',
            dashed_word,
        ):
            return True

        return None

    def dehyphen_dashed_word(self, dash: str) -> str:  # pylint: disable=too-many-return-statements
        """Remove hyphen from word if rules are satisfied"""
        compound_word: str = re.sub('- ', '', dash)
        dashed_word: str = re.sub('- ', '-', dash)

        _compound_word: str = compound_word.lower()
        _dashed_word: str = dashed_word.lower()

        if self.is_whitelisted(_compound_word):
            return compound_word

        if self.is_whitelisted(_dashed_word):
            return dashed_word

        if self.is_hyphenated_compound(dashed_word):
            return self.add_to_whitelist(dashed_word, WhitelistReason.HyphenatedCompound)

        _compound_word_frequency: int = self.word_frequencies.get(_compound_word, 0)
        _dashed_word_frequency: int = self.word_frequencies.get(_dashed_word, 0)

        if _compound_word_frequency > _dashed_word_frequency:
            return self.add_to_whitelist(compound_word, WhitelistReason.Frequency)

        if _dashed_word_frequency > _compound_word_frequency:
            return self.add_to_whitelist(dashed_word, WhitelistReason.Frequency)

        if _dashed_word_frequency > 0:
            self.unresolved.add(dash)
            return dash

        left_word, right_word = dashed_word.split('-')

        if (
            not self.is_whitelisted(left_word)
            and not self.is_whitelisted(right_word)
            and self.word_frequencies.get(left_word, 0) == 0
        ):
            return self.add_to_whitelist(compound_word, WhitelistReason.UnknownParts)

        self.unresolved.add(dash)

        return dash

    def dehyphen_text(self, text: str) -> str:
        """Remove dehyphens in text"""
        text: str = re.sub(r'\n{3,}', r'\n\n', text)

        # add paragraph markers:
        text = re.sub(r'\n\n', PARAGRAPH_MARKER, text)

        # remove paragraph marker if previous line is ELH (end-of-line hyphenation)
        text = re.sub(rf'-\s*{PARAGRAPH_MARKER}', '- ', text)

        # normalize all white spaces to a single space
        text = ' '.join(text.split())

        dashed_words: set[str] = find_dashed_words(text)

        for dashed_word in dashed_words:
            dehyphened_word = self.dehyphen_dashed_word(dashed_word)
            if dehyphened_word != dashed_word:
                text = re.sub(dashed_word, dehyphened_word, text)

        text = text.strip()
        text = re.sub(PARAGRAPH_MARKER, '\n\n', text)

        text = merge_paragraphs(text, self.paragraph_merge_strategy)

        return text

    def flush(self) -> None:
        store_token_set(self.whitelist, self.whitelist_filename)
        store_token_set(self.unresolved, self.unresolved_filename)
        store_dict(self.whitelist_log, self.whitelist_log_filename)

    def load_state(self) -> None:
        self.whitelist = load_token_set(self.whitelist_filename)
        self.whitelist_log = load_dict(self.whitelist_log_filename)
        self.unresolved = load_token_set(self.unresolved_filename)

    @staticmethod
    def create_dehypen(data_folder: str, word_frequencies: str | dict = None) -> Callable[[str], str]:
        """Create a dehyphen service. Return wrapped dehyphen function."""
        logger.info(f"dehyphen path: {data_folder}")
        dehyphenator: SwedishDehyphenator = SwedishDehyphenator(
            data_folder=data_folder, word_frequencies=word_frequencies
        )

        def dehyphen(text: str) -> str:
            """Remove hyphens from `text`."""
            dehyphenated_text = dehyphenator.dehyphen_text(text)

            return dehyphenated_text

        return dehyphen


def merge_paragraphs(text: str, paragraph_merge_strategy: ParagraphMergeStrategy) -> str:
    """Merge paragraphs"""
    if paragraph_merge_strategy == ParagraphMergeStrategy.MergeIfWordsOnlySeparatedByTwoNewlines:
        return re.sub(r"(\w+)(\n\n)(\w+)", r"\1 \3", text)

    if paragraph_merge_strategy == ParagraphMergeStrategy.MergeAll:
        return re.sub('\n\n', ' ', text)

    return text
