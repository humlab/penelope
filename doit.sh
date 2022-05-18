#!/bin/bash

set -e

CORPUS_FOLDER=/data/westac
TARGET_FOLDER=/data/westac/shared

CORPUS_FILENAME=riksdagens-protokoll.1920-2019.sparv4.csv.zip
CONFIG_FILENAME='./doit.yml'
#CORPUS_FILENAME=riksdagens-protokoll.1920-2019.test.sparv4.csv.zip
#CONFIG_FILENAME='./tests/test_data/riksdagens-protokoll.yml'

RUN_OPTS="--tf-threshold 10 --tf-threshold-mask  --lemmatize --to-lower --partition-key year --enable-checkpoint"
# --force-checkpoint

run()
{
    concept_word=$1
    pos_includes=$2
    pos_paddings=$3
    context_width=$4

    basename="NEW_${concept_word}_w${context_width}_${pos_includes//|}_${pos_paddings//|}_LEMMA_KEEPSTOPS"
    mkdir -p  ${TARGET_FOLDER}/$basename

    command="poetry run python penelope/scripts/co_occurrence.py $RUN_OPTS --pos-includes "$pos_includes" --pos-paddings "$pos_paddings" --context-width $context_width --concept $concept_word \
         ${CONFIG_FILENAME} \
         ${CORPUS_FOLDER}/${CORPUS_FILENAME} \
         ${TARGET_FOLDER}/${basename}/${basename}"

    echo $command > ${TARGET_FOLDER}/${basename}/run_command.sh

    $command
}

CONCEPTS="$*"

for concept in $CONCEPTS ; do

    #run $concept "VB" "PASSTHROUGH" 1
    run $concept "JJ" "PASSTHROUGH" 1
    run $concept "NN|PM" "PASSTHROUGH" 5
    run $concept "NN|PM" "PASSTHROUGH" 10

done
