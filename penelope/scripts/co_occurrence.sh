#!/bin/bash

set -e

PREFIX=1920-2019_
CORPUS_FOLDER=/data/westac
TARGET_FOLDER=/data/westac/shared

CORPUS_FILENAME=riksdagens-protokoll.1920-2019.sparv4.csv.zip
CONFIG_FILENAME='./doit.yml'
TF_THRESHOLD=10

run()
{
    concept_word=$1
    pos_includes=$2
    pos_paddings=$3
    context_width=$4

    command_opts=" --tf-threshold ${TF_THRESHOLD} --tf-threshold-mask --ignore-padding --keep-numerals --enable-checkpoint --lemmatize --to-lowercase --partition-key year "
    command_opts=" ${command_opts} --pos-includes $pos_includes --pos-paddings $pos_paddings "
    command_opts=" ${command_opts} --context-width $context_width "

    if [ "$concept_word" != "x" ]; then
        command_opts=" ${command_opts} --concept $concept_word "
    fi

    basename="${PREFIX}${concept_word}_w${context_width}_${pos_includes//|}_${pos_paddings//|}_TF${TF_THRESHOLD}_LEMMA_KEEPSTOPS"

    mkdir -p  ${TARGET_FOLDER}/$basename

    output_folder="${TARGET_FOLDER}/${basename}/${basename}"
    input_filename="${CORPUS_FOLDER}/${CORPUS_FILENAME}"
    command="poetry run co_occurrence $command_opts ${CONFIG_FILENAME} $input_filename $output_folder"

    echo $command > ${TARGET_FOLDER}/${basename}/run_command.sh

    $command
}

CONCEPTS="$*"

for concept in $CONCEPTS ; do

    run $concept "VB" "PASSTHROUGH" 1 
    run $concept "JJ" "PASSTHROUGH" 1
    run $concept "NN|PM" "PASSTHROUGH" 5
    run $concept "NN|PM" "PASSTHROUGH" 10

done
