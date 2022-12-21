#!/bin/bash

default_opts="./run-opts.yml"
target_folder="./data/inidun/courier/co_occurrence"
tag_prefix="Courier"
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

# # 1) concept "culture",       ignore concept, adjectives, nouns, verbs, w5,  ft10, nolemma, only alpha:
# # 2) concept "culture",       ignore concept, adjectives, nouns, verbs, w11, ft10, nolemma, only alpha:
# # 3) concept "cultures",      ignore concept, adjectives, nouns, verbs, w5,  ft10, nolemma, only alpha:
# # 4) concept "cultures",      ignore concept, adjectives, nouns, verbs, w11, ft10, nolemma, only alpha:
# # 5) concept "civilization",  ignore concept, adjectives, nouns, verbs, w5,  ft10, nolemma, only alpha:
# # 6) concept "civilization",  ignore concept, adjectives, nouns, verbs, w11, ft10, nolemma, only alpha:
# # 7) concept "civilizations", ignore concept, adjectives, nouns, verbs, w5,  ft10, nolemma, only alpha:
# # 8) concept "civilizations", ignore concept, adjectives, nouns, verbs, w11, ft10, nolemma, only alpha:

run "culture"       2 10
run "culture"       5 10
run "cultures"      2 10
run "cultures"      5 10
run "civilization"  2 10
run "civilization"  5 10
run "civilizations" 2 10
run "civilizations" 5 10


# # Courier corpus (page)
# # Co-occurrence computations

# co_occurrence $DEFAULT_OPTS --context-width 2 --concept "culture" --tf-threshold 10 doit.yml /data/inidun/courier_page_20210921.zip /data/inidun/shared/Courier_culture_w5_nvadj_nolemma/Courier_culture_w5_nvadj_nolemma_co-occurrence.csv.zip Courier_culture_w5_nvadj_nolemma
# co_occurrence $DEFAULT_OPTS --context-width 5 --concept "culture" --tf-threshold 10 doit.yml /data/inidun/courier_page_20210921.zip /data/inidun/shared/Courier_culture_w11_nvadj_nolemma/Courier_culture_w11_nvadj_nolemma_co-occurrence.csv.zip Courier_culture_w11_nvadj_nolemma
# co_occurrence $DEFAULT_OPTS --context-width 2 --concept "cultures" --tf-threshold 10 doit.yml /data/inidun/courier_page_20210921.zip /data/inidun/shared/Courier_cultures_w5_nvadj_nolemma/Courier_cultures_w5_nvadj_nolemma_co-occurrence.csv.zip Courier_cultures_w5_nvadj_nolemma
# co_occurrence $DEFAULT_OPTS --context-width 5 --concept "cultures" --tf-threshold 10 doit.yml /data/inidun/courier_page_20210921.zip /data/inidun/shared/Courier_cultures_w11_nvadj_nolemma/Courier_cultures_w11_nvadj_nolemma_co-occurrence.csv.zip Courier_cultures_w11_nvadj_nolemma
# co_occurrence $DEFAULT_OPTS --context-width 2 --concept "civilization" --tf-threshold 10 doit.yml /data/inidun/courier_page_20210921.zip /data/inidun/shared/Courier_civilization_w5_nvadj_nolemma/Courier_civilization_w5_nvadj_nolemma_co-occurrence.csv.zip Courier_civilization_w5_nvadj_nolemma
# co_occurrence $DEFAULT_OPTS --context-width 5 --concept "civilization" --tf-threshold 10 doit.yml /data/inidun/courier_page_20210921.zip /data/inidun/shared/Courier_civilization_w11_nvadj_nolemma/Courier_civilization_w11_nvadj_nolemma_co-occurrence.csv.zip Courier_civilization_w11_nvadj_nolemma
# co_occurrence $DEFAULT_OPTS --context-width 2 --concept "civilizations" --tf-threshold 10 doit.yml /data/inidun/courier_page_20210921.zip /data/inidun/shared/Courier_civilizations_w5_nvadj_nolemma/Courier_civilizations_w5_nvadj_nolemma_co-occurrence.csv.zip Courier_civilizations_w5_nvadj_nolemma
# co_occurrence $DEFAULT_OPTS