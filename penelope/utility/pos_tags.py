from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Container, Dict, List, Union

import numpy as np
import pandas as pd
from loguru import logger
from more_itertools import collapse

SUC_tags = {
    'AB': 'Adverb',
    'DT': 'Determinator',
    'HA': 'Adverbs for inquire',
    'HD': 'Determinator for inquire',
    'HP': 'Pronoun for inquire',
    'HS': 'Pronoun for inquire',
    'IE': 'Infinitive mark',
    'IN': 'Interjection',
    'JJ': 'Adjectiv',
    'KN': 'Conjunction',
    'NN': 'Noun',
    'PC': 'Participle',
    'PL': 'Particle',
    'PM': 'Proper noun',
    'PN': 'Pronoun',
    'PP': 'Preposition',
    'PS': 'Possesive pronoun',
    'RG': 'Numeral',
    'RO': 'Numeral',
    'SN': 'Subjuncion',
    'UO': 'Foreign ord',
    'VB': 'Verb',
    'MAD': 'Delimiter',
    'MID': 'Delimiter',
    'PAD': 'Delimiter',
}

SUC_PoS_tag_groups = {
    'Pronoun': ['DT', 'HD', 'HP', 'HS', 'PS', 'PN'],
    'Noun': ['NN', 'PM'],
    'Verb': ['VB'],
    'Adverb': ['AB', 'HA', 'IE', 'IN', 'PL'],
    'Numeral': ['RG', 'RO'],
    'Adjective': ['JJ'],
    'Preposition': ['PP'],
    'Conjunction': ['KN', 'SN'],
    'Delimiter': ['MAD', 'MID', 'PAD'],
    'Other': ['PC', 'UO'],
}


PD_PoS_tag_groups = pd.DataFrame(
    data={
        'tag_group_name': [
            'Pronoun',
            'Noun',
            'Verb',
            'Adverb',
            'Numeral',
            'Adjective',
            'Preposition',
            'Conjunction',
            'Delimiter',
            'Other',
        ],
        'tag_group_id': list(range(1, 11)),
    }
).set_index('tag_group_name')

PD_SUC_PoS_tags = (
    pd.DataFrame(
        data=[
            ('AB', 'Adverb', 'Adverb'),
            ('DT', 'Pronoun', 'Determinator'),
            ('HA', 'Adverb', 'Adverbs (inq.)'),
            ('HD', 'Pronoun', 'Det. (inq.)'),
            ('HP', 'Pronoun', 'Pronoun (inq.)'),
            ('HS', 'Pronoun', 'Pronoun (inq.)'),
            ('IE', 'Adverb', 'Inf. mark'),
            ('IN', 'Adverb', 'Interjection'),
            ('JJ', 'Adjective', 'Adjectiv'),
            ('KN', 'Conjunction', 'Conjunction'),
            ('NN', 'Noun', 'Noun'),
            ('PC', 'Verb', 'Participle'),
            ('PL', 'Adverb', 'Particle'),
            ('PM', 'Noun', 'Proper noun'),
            ('PN', 'Pronoun', 'Pronoun'),
            ('PP', 'Preposition', 'Preposition'),
            ('PS', 'Pronoun', 'Poss. pron.'),
            ('RG', 'Numeral', 'Numeral'),
            ('RO', 'Numeral', 'Numeral'),
            ('SN', 'Conjunction', 'Subjuncion'),
            ('UO', 'Other', 'Foreign ord'),
            ('VB', 'Verb', 'Verb'),
            ('MAD', 'Delimiter', 'Delimiter'),
            ('MID', 'Delimiter', 'Delimiter'),
            ('PAD', 'Delimiter', 'Delimiter'),
        ],
        columns=['tag', 'tag_group_name', 'description'],
    )
    .rename_axis('pos_id')
    .reset_index()
    .set_index('tag')
    .assign(tag=lambda x: x.index)
)

PD_Universal_PoS_tags = (
    pd.DataFrame(
        data=[
            ('ADJ', 'Adjective', 'Adjective'),
            ('ADP', 'Preposition', 'Preposition'),
            ('ADV', 'Adverb', 'Adverb'),
            ('AUX', 'Other', 'Other'),
            ('CONJ', 'Conjunction', 'Conjunction'),
            ('CCONJ', 'Conjunction', 'Conjunction, coordinating'),
            ('DET', 'Pronoun', 'Determinator'),
            ('INTJ', 'Adverb', 'Interjection'),
            ('NOUN', 'Noun', 'Noun'),
            ('NUM', 'Numeral', 'Numeral'),
            ('PART', 'Adverb', ''),
            ('PRON', 'Pronoun', 'Pronoun'),
            ('PROPN', 'Noun', 'Proper Noun'),
            ('PUNCT', 'Delimiter', 'Punctuation'),
            ('SCONJ', 'Conjunction', 'Subjuncion'),
            ('SYM', 'Other', 'Symbol'),
            ('VERB', 'Verb', 'Verb'),
            ('X', 'Other', 'Other'),
            ('EOL', 'Delimiter', 'End of line'),
            ('SPACE', 'Delimiter', 'Space'),
        ],
        columns=['tag', 'tag_group_name', 'description'],
    )
    .rename_axis('pos_id')
    .reset_index()
    .set_index('tag')
    .assign(tag=lambda x: x.index)
)


PD_PennTree_O5_PoS_tags = (
    pd.DataFrame(
        data=[
            ('$', 'Other', 'SYM', 'symbol, currency'),
            ('``', 'Delimiter', 'PUNCT', 'opening quotation mark'),
            ("''", 'Delimiter', 'PUNCT', 'closing quotation mark'),
            (',', 'Delimiter', 'PUNCT', 'punctuation mark, comma'),
            ('-LRB-', 'Delimiter', 'PUNCT', 'left round bracket'),
            ('-RRB-', 'Delimiter', 'PUNCT', 'right round bracket'),
            ('.', 'Delimiter', 'PUNCT', 'punctuation mark, sentence closer'),
            (':', 'Delimiter', 'PUNCT', 'punctuation mark, colon or ellipsis'),
            ('ADD', 'Other', 'X', 'email'),
            ('AFX', 'Adjective', 'ADJ', 'affix'),
            ('CC', 'Conjunction', 'CCONJ', 'conjunction, coordinating'),
            ('CD', 'Numeral', 'NUM', 'cardinal number'),
            ('DT', 'Pronoun', 'DET', 'determiner'),
            ('EX', 'Pronoun', 'PRON', 'existential there'),
            ('FW', 'Other', 'X', 'foreign word'),
            ('GW', 'Other', 'X', 'additional word in multi-word expression'),
            ('HYPH', 'Delimiter', 'PUNCT', 'punctuation mark, hyphen'),
            ('IN', 'Preposition', 'ADP', 'conjunction, subordinating or preposition'),
            ('JJ', 'Adjective', 'ADJ', 'adjective'),
            ('JJR', 'Adjective', 'ADJ', 'adjective, comparative'),
            ('JJS', 'Adjective', 'ADJ', 'adjective, superlative'),
            ('LS', 'Other', 'X', 'list item marker'),
            ('MD', 'Verb', 'VERB', 'verb, modal auxiliary'),
            ('NFP', 'Delimiter', 'PUNCT', 'superfluous punctuation'),
            ('NIL', 'Other', 'X', 'missing tag'),
            ('NN', 'Noun', 'NOUN', 'noun, singular or mass'),
            ('NNP', 'Noun', 'PROPN', 'noun, proper singular'),
            ('NNPS', 'Noun', 'PROPN', 'noun, proper plural'),
            ('NNS', 'Noun', 'NOUN', 'noun, plural'),
            ('PDT', 'Pronoun', 'DET', 'predeterminer'),
            ('POS', 'Adverb', 'PART', 'possessive ending'),
            ('PRP', 'Pronoun', 'PRON', 'pronoun, personal'),
            ('PRP$', 'Pronoun', 'DET', 'pronoun, possessive'),
            ('RB', 'Adverb', 'ADV', 'adverb'),
            ('RBR', 'Adverb', 'ADV', 'adverb, comparative'),
            ('RBS', 'Adverb', 'ADV', 'adverb, superlative'),
            ('RP', 'Adverb', 'ADP', 'adverb, particle'),
            ('SP', 'Delimiter', 'SPACE', 'space'),
            ('SYM', 'Other', 'SYM', 'symbol'),
            ('TO', 'Adverb', 'PART', 'infinitival “to”'),
            ('UH', 'Adverb', 'INTJ', 'interjection'),
            ('VB', 'Verb', 'VERB', 'verb, base form'),
            ('VBD', 'Verb', 'VERB', 'verb, past tense'),
            ('VBG', 'Verb', 'VERB', 'verb, gerund or present participle'),
            ('VBN', 'Verb', 'VERB', 'verb, past participle'),
            ('VBP', 'Verb', 'VERB', 'verb, non-3rd person singular present'),
            ('VBZ', 'Verb', 'VERB', 'verb, 3rd person singular present'),
            ('WDT', 'Pronoun', 'DET', 'wh-determiner'),
            ('WP', 'Pronoun', 'PRON', 'wh-pronoun, personal'),
            ('WP$', 'Pronoun', 'DET', 'wh-pronoun, possessive'),
            ('WRB', 'Adverb', 'ADV', 'wh-adverb'),
            ('XX', 'Other', 'X', 'unknown'),
            ('_SP', 'Delimiter', 'SPACE', ''),
        ],
        columns=['tag', 'tag_group_name', 'universal_tag', 'description'],
    )
    .rename_axis('pos_id')
    .reset_index()
    .set_index('tag')
    .assign(tag=lambda x: x.index)
)


class PoS_Tag_Scheme:
    def __init__(self, df: pd.DataFrame) -> None:
        self.PD_PoS_tags: pd.DataFrame = df
        self.PD_PoS_groups: pd.DataFrame = df.groupby('tag_group_name')['tag'].agg(list)
        self.pos_to_id: dict = df['pos_id'].to_dict()
        self.id_to_pos: dict = {v: k for k, v in self.pos_to_id.items()}
        self.groups: Dict[str, List[str]] = self.PD_PoS_groups.to_dict()
        self.tag_to_group: Dict[str, str] = self.PD_PoS_tags.set_index('tag')['tag_group_name'].to_dict()

    @property
    def tags(self) -> List[str]:

        return list(self.PD_PoS_tags.index)

    def unwrap(self, x: Any) -> Any:
        if isinstance(x, str):
            if x in self.groups:
                return self.groups[x]
            if '|' in x:
                return str_to_pos_tags(x)
            return x
        return list(x)

    def exclude(self, excludes: Union[str, Container[str]] = None) -> List[str]:

        _all_tags = list(self.PD_PoS_tags.index)

        if excludes is None:
            return _all_tags

        if isinstance(excludes, str):
            excludes = [excludes]

        excludes = set(collapse(map(self.unwrap, excludes)))

        return [x for x in _all_tags if x not in excludes]

    def all_types_except(self, tags: Union[str, Container[str]] = None) -> List[str]:

        return self.exclude([tags, self.Delimiter])

    @property
    def Pronoun(self) -> List[str]:
        return self.groups.get('Pronoun', [])

    @property
    def Noun(self) -> List[str]:
        return self.groups.get('Noun', [])

    @property
    def Verb(self) -> List[str]:
        return self.groups.get('Verb', [])

    @property
    def Adverb(self) -> List[str]:
        return self.groups.get('Adverb', [])

    @property
    def Numeral(self) -> List[str]:
        return self.groups.get('Numeral', [])

    @property
    def Adjective(self) -> List[str]:
        return self.groups.get('Adjective', [])

    @property
    def Preposition(self) -> List[str]:
        return self.groups.get('Preposition', [])

    @property
    def Conjunction(self) -> List[str]:
        return self.groups.get('Conjunction', [])

    @property
    def Delimiter(self) -> List[str]:
        return self.groups.get('Delimiter', [])

    @property
    def Other(self) -> List[str]:
        return self.groups.get('Other', [])

    @property
    def description(self) -> Dict[str, str]:
        return self.PD_PoS_tags.set_index('tag')['description'].to_dict()

    def PoS_group_counts(self, PoS_sequence: pd.Series) -> dict:
        """Computes word counts (total and per part-of-speech) given tagged_frame"""

        if PoS_sequence is None or len(PoS_sequence) == 0:
            return {}

        if not isinstance(PoS_sequence, pd.Series):
            raise ValueError(f"Expected pd.Series, found {type(PoS_sequence)}")

        tag_counts = defaultdict(int)
        for PoS in PoS_sequence:
            tag_counts[PoS] += 1

        """Convert to strings if PoS-sequence is integers"""
        if np.issubdtype(PoS_sequence.dtype, np.integer):
            ig = self.id_to_pos.get
            tag_counts = {ig(k, 'XYZ'): v for k, v in tag_counts.items()}

        """Sum up into PoS-groups"""
        group_counts = {k: 0 for k in self.groups.keys()}
        tg = self.tag_to_group.get
        n_tokens: int = 0
        for k, v in tag_counts.items():
            group_name: str = tg(k)
            if group_name:
                group_counts[group_name] += v
            else:
                logger.error(f"skipped {v} tokens in tagged_frame_to_PoS_group_counts with unknown PoS tag {k}")
            if group_name != 'Delimiter':
                n_tokens += v

        group_counts.update(n_raw_tokens=n_tokens, n_tokens=n_tokens)

        return group_counts


Known_PoS_Tag_Schemes = dict(
    SUC=PoS_Tag_Scheme(PD_SUC_PoS_tags),
    Universal=PoS_Tag_Scheme(PD_Universal_PoS_tags),
    PennTree=PoS_Tag_Scheme(PD_PennTree_O5_PoS_tags),
)


@dataclass
class PoS_Tag_Schemes:

    SUC: PoS_Tag_Scheme = PoS_Tag_Scheme(PD_SUC_PoS_tags)
    Universal: PoS_Tag_Scheme = PoS_Tag_Scheme(PD_Universal_PoS_tags)
    PennTree: PoS_Tag_Scheme = PoS_Tag_Scheme(PD_PennTree_O5_PoS_tags)


PoS_TAGS_SCHEMES = PoS_Tag_Schemes()

Known_PoS_Tag_Schemes = asdict(PoS_TAGS_SCHEMES)


def get_pos_schema(name: str) -> PoS_Tag_Scheme:
    return Known_PoS_Tag_Schemes.get(name, None)


def pos_tags_to_str(tags: List[str]) -> str:
    return f"|{'|'.join(tags)}|"


def str_to_pos_tags(tags: str) -> List[str]:
    return tags.strip('|').split('|')
