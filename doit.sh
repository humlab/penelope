#!/bin/bash

# Beställningen kommer här:

# Vi är intresserade av två concept: information respektive propaganda – lemmatiserade. I båda fallen vill ha concept kvar. vi vill kolla på:

# SUC tags:
#     'Pronoun': ['DT', 'HD', 'HP', 'HS', 'PS', 'PN'],
#     'Noun': ['NN', 'PM', 'UO'],
#     'Verb': ['PC', 'VB'],
#     'Adverb': ['AB', 'HA', 'IE', 'IN', 'PL'],
#     'Numeral': ['RG', 'RO'],
#     'Adjective': ['JJ'],
#     'Preposition': ['PP'],
#     'Conjunction': ['KN', 'SN'],
#     'Delimiter': ['MAD', 'MID', 'PAD'],

run()
{
    concept_word=$1
    pos_classes=$2
    context_width=$3

    basename="${concept_word}_w${context_width}_${pos_classes//|}_lemma_no_stops"
    mkdir -p /data/westac/shared/$basename

    echo "co_occurrence --pos-includes $pos_classes --context-width $context_width --lemmatize --to-lowercase --partition-key year \
        --remove-stopwords swedish --concept $concept_word \
        resources/riksdagens-protokoll.yml  \
        /data/westac/data/riksdagens-protokoll.1920-2019.sparv4.csv.zip \
        /data/westac/shared/${basename}/${basename}"
}

CONCEPTS="information propaganda"

for concept in $CONCEPTS ; do

    # samförekommande substantiv i dels ett 11-ordsfönster, dels ett 7-ordsfönster.
    run $concept "|NN|PM|UO|" 11
    run $concept "|NN|PM|UO|" 7

    # samförekommande adjektiv i ett 3-ordsfönster
    run $concept "|JJ|" 3

    # samförekommande verb i ett 3-ordsfönster
    run $concept "|PC|VB|" 3

done

