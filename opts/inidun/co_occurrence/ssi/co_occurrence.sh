#!/bin/bash

default_opts="./run-opts.yml"
target_folder="./data"
tag_prefix="SSI_nonregional"
tag_suffix="nvadj_nolemma"

function run()
{
    concept="$1"
    context_width=$2
    window_width=$(($context_width * 2 + 1))
    tf_threshold=$3
    tag="${tag_prefix}_${concept}_w${window_width}_tf${tf_threshold}_${tag_suffix}"
    co-occurrence --options-filename $default_opts \
        --context-width ${context_width} \
        --concept "${concept}" \
        --tf-threshold ${tf_threshold} \
        --output-filename $target_folder/${tag}/${tag}_co-occurrence.csv.zip
}

run "culture" 3 1
run "cultures" 3 1
run "civilization" 3 1
run "civilizations" 3 1

# 1)   concept “culture”,       ignore concept, adjectives, nouns, verbs, w7, ft1, nolemma, only alpha:
# 2)   concept “cultures”,      ignore concept, adjectives, nouns, verbs, w7, ft1, nolemma, only alpha:
# 3)   concept “civilization”,  ignore concept, adjectives, nouns, verbs, w7, ft1, nolemma, only alpha:
# 4)   concept “civilizations”, ignore concept, adjectives, nouns, verbs, w7, ft1, nolemma, only alpha:
