#!/bin/bash

set -e
export PYTHONPATH=.

PREFIX=1970_
CORPUS_FOLDER=/data/westac
# CORPUS_FILENAME=riksdagens-protokoll.1867-2019.sparv4.csv.zip
CORPUS_FILENAME=riksdagens-protokoll.1920-2019.9files.sparv4.csv.zip
TARGET_FOLDER=../data

TIMESTAMP=$(date '+%Y%m%d')
CONFIG_FILENAME="kb-labb_corpus_config_${TIMESTAMP}.yml"
OPTIONS_FILENAME="kb-labb_options_${TIMESTAMP}.yml"

TF_THRESHOLD=20

generate_corpus_config()
{
    cat << EOF > /tmp/${CONFIG_FILENAME}
corpus_name: riksdagens-protokoll
corpus_pattern: '*sparv4.csv.zip'
corpus_type: 3
language: swedish
checkpoint_opts:
    content_type_code: 1
    sep: "\t"
    quoting: 3
    document_index_name: null
    document_index_sep: null
    text_column: token
    lemma_column: baseform
    pos_column: pos
    extra_columns: []
    custom_serializer_classname: penelope.pipeline.sparv.convert.SparvCsvSerializer
    deserialize_processes: 8
    deserialize_chunksize: 4
    index_column: null
pipelines:
  tagged_frame_pipeline: penelope.pipeline.sparv.pipelines.to_tagged_frame_pipeline
pipeline_payload:
  source: ${CORPUS_FILENAME}
  document_index_source: null
  document_index_sep: "\t"
  filenames: null
  memory_store:
    lang: se
    tagger: Sparv
    sparv_version: 4
    text_column: token
    lemma_column: baseform
    pos_column: pos
  pos_schema_name: SUC
filter_opts:
text_reader_opts:
  as_binary: false
  filename_fields:
   - "year:prot\\\\_(\\\\d{4}).*"
   - "year2:prot_\\\\d{4}(\\\\d{2})__*"
   - "number:prot_\\\\d+[afk_]{0,4}__(\\\\d+).*"
  filename_filter: null
  filename_pattern: '*.csv'
  index_field: null
  sep: "\t"
  quoting: 3
EOF
}

generate_default_options_file()
{

    cat << EOF > /tmp/${OPTIONS_FILENAME}
append_pos: false
compute_chunk_size: 10
compute_processes: null
context_width: 1
create_subfolder: true
deserialize_processes: 4
enable_checkpoint: true
filename_pattern: null
force_checkpoint: false
ignore_concept: false
ignore_padding: true
keep_numerals: true
keep_symbols: true
lemmatize: true
max_word_length: null
min_word_length: 1
only_alphabetic: false
only_any_alphanumeric: false
partition_key: !!python/tuple
- year
phrase: !!python/tuple []
phrase_file: null
pos_excludes: ''
pos_includes: ''
pos_paddings: PASSTHROUGH
remove_stopwords: null
tf_threshold: 10
tf_threshold_mask: true
to_lower: true
EOF
}

run()
{
    concept_word=$1
    pos_includes=$2
    pos_paddings=$3
    context_width=$4

    command_opts="\
        --tf-threshold ${TF_THRESHOLD} \
        --tf-threshold-mask \
        --pos-includes $pos_includes \
        --pos-paddings $pos_paddings \
        --context-width $context_width
    "

    if [ "$concept_word" != "x" ]; then
        command_opts=" ${command_opts} --concept $concept_word "
    fi

    basename="${PREFIX}${concept_word}_w${context_width}_${pos_includes//|}_${pos_paddings//|}_TF${TF_THRESHOLD}_LEMMA_KEEPSTOPS"

    output_folder="${TARGET_FOLDER}/${basename}/${basename}"
    input_filename="${CORPUS_FOLDER}/${CORPUS_FILENAME}"

    command="python penelope/scripts/co_occurrence.py --options-filename /tmp/${OPTIONS_FILENAME} $command_opts /tmp/${CONFIG_FILENAME} $input_filename $output_folder"

    mkdir -p ${TARGET_FOLDER}/$basename

    echo $command > ${TARGET_FOLDER}/${basename}/run_command.sh

    $command

}

CONCEPTS="$*"

generate_corpus_config
generate_default_options_file

for concept in $CONCEPTS ; do

    run $concept "VB" "PASSTHROUGH" 1
    # run $concept "JJ" "PASSTHROUGH" 1
    # run $concept "NN|PM" "PASSTHROUGH" 5
    # run $concept "NN|PM" "PASSTHROUGH" 10

done
