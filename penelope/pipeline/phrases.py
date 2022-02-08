import os
from typing import List, Tuple

import pandas as pd
from loguru import logger

from penelope.corpus.readers import PhraseSubstitutions

PHRASE_PAD: str = "(*)"


def detect_phrases(
    target_series: pd.Series,
    phrases: PhraseSubstitutions,
    ignore_case: str = False,
) -> List[Tuple[int, str, int]]:
    """Detects and updates phrases on document `doc`.

    Args:
        phrases (List[List[str]]): [description]
        doc (pd.DataFrame): [description]
        target (str): [description]
    """

    if phrases is None:
        return []

    if not isinstance(phrases, (list, dict)):
        raise TypeError("phrase must be dict ot list")

    phrases = (
        {'_'.join(phrase): phrase for phrase in phrases}
        if isinstance(phrases, list)
        else {token.replace(' ', ''): phrase for token, phrase in phrases.items()}
    )

    if ignore_case:
        phrases = {key: [x.lower() for x in phrase] for key, phrase in phrases.items()}

    found_phrases = []
    for replace_token, phrase in phrases.items():

        if len(phrase) < 2:
            continue

        for idx in target_series[target_series == phrase[0]].index:

            if (target_series[idx : idx + len(phrase)] == phrase).all():
                found_phrases.append((idx, replace_token, len(phrase)))

    return found_phrases


def merge_phrases(
    doc: pd.DataFrame,
    phrase_positions: List[Tuple[int, List[str]]],
    target_column: str,
    pad: str = "*",
) -> pd.DataFrame:
    """Returns (same) document with found phrases merged into a single token.
    The first word in phrase is replaced by entire phrase, and consequtive words are replaced by `pad`.
    Note that the phrase will have the same PoS tag as the first word."""
    for idx, token, n in phrase_positions:
        doc.loc[idx, target_column] = token
        doc.loc[idx + 1 : idx + n - 1, target_column] = pad
        # doc.loc[idx+1:len(phrase) + 1, pos_column] = 'MID'
    return doc


def parse_phrases(phrase_file: str, phrases: List[str]):

    try:
        phrase_specification = {}

        if phrases:
            phrases = [p.split() for p in phrases]
            phrase_specification.update({'_'.join(phrase).replace(' ', '_'): phrase for phrase in (phrases or [])})

        if phrase_file:
            """Expect file to be lines with format:
            ...
            replace_string; the phrase to replace
            ...
            """
            if os.path.isfile(phrase_file):
                with open(phrase_file, "r") as fp:
                    data_str: str = fp.read()
            else:
                data_str: str = phrase_file

            phrase_lines: List[str] = [line for line in data_str.splitlines() if line.strip() != ""]

            phrase_specification.update(
                {
                    key.strip().replace(' ', '_'): phrase.strip().split()
                    for key, phrase in [line.split(";") for line in phrase_lines]
                }
            )

        if len(phrase_specification) == 0:
            return None

        return phrase_specification

    except Exception as ex:
        logger.error(ex)
        raise ValueError("failed to decode phrases. please review file and/or arguments") from ex
