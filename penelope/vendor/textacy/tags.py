PoS_tags = {
    'ADJ',
    'ADP',
    'ADV',
    'AUX',
    'CONJ',
    'CCONJ',
    'DET',
    'INTJ',
    'NOUN',
    'NUM',
    'PART',
    'PRON',
    'PROPN',
    'PUNCT',
    'SCONJ',
    'SYM',
    'VERB',
    'X',
    'EOL',
    'SPACE',
}

# FIXME: #11 Check textaCy PoS groups
PoS_tag_groups = {
    'Pronoun': ['DET', 'PRON'],
    'Noun': ['NOUN', 'PROPN'],
    'Verb': ['VERB'],
    'Adverb': ['ADV', 'INTJ', 'PART'],
    'Numeral': ['NUM'],
    'Adjective': ['ADJ'],
    'Preposition': ['ADP'],
    'Conjunction': ['CONJ', 'CCONJ', 'SCONJ'],
}
