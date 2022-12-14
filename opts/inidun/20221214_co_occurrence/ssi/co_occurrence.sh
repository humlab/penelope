#!/bin/bash

default_opts="./default-run-opts.yml"

function run()
{
    concept="$1"
    context_width=$2
    tf_threshold=$3
    tag=$4
    target_folder=$5
    co-occurrence --options-filename $default_opts \
        --context-width ${context_width} \
        --concept "${concept}" \
        --tf-threshold ${tf_threshold} \
        --output-filename $target_folder/${tag}/${tag}_co-occurrence.csv.zip
}

run "culture" 3 1 "SSI_${culture}_w7_tf1_nvadj_nolemma" "./data/inidun/ssi/co_occurrence"

# 1)   concept “culture”, ignore concept, adjectives, nouns, verbs, w7, ft1, nolemma, only alpha:
# co-occurrence $DEFAULT_OPTS --context-width 3 --concept "culture" $CONFIG_FILE $CORPUS_FILE $TARGET_FOLDER/SSI_culture_w7_ft1_nvadj_nolemma/SSI_culture_w7_ft1_nvadj_nolemma_co-occurrence.csv.zip
# 2)   concept “cultures”, ignore concept, adjectives, nouns, verbs, w7, ft1, nolemma, only alpha:
# co_occurrence $DEFAULT_OPTS --context-width 3 --concept "cultures" $CONFIG_FILE $CORPUS_FILE $TARGET_FOLDER/SSI_cultures_w7_ft1_nvadj_nolemma/SSI_cultures_w7_ft1_nvadj_nolemma_co-occurrence.csv.zip SSI_cultures_w7_ft1_nvadj_nolemma
# # 3)   concept “civilization”, ignore concept, adjectives, nouns, verbs, w7, ft1, nolemma, only alpha:
# co_occurrence $DEFAULT_OPTS --context-width 3 --concept "civilization" $CONFIG_FILE $CORPUS_FILE $TARGET_FOLDER/SSI_civilization_w7_ft1_nvadj_nolemma/SSI_civilization_w7_ft1_nvadj_nolemma_co-occurrence.csv.zip SSI_civilization_w7_ft1_nvadj_nolemma
# # 4)   concept “civilizations”, ignore concept, adjectives, nouns, verbs, w7, ft1, nolemma, only alpha:
# co_occurrence $DEFAULT_OPTS --context-width 3 --concept "civilizations" $CONFIG_FILE $CORPUS_FILE $TARGET_FOLDER/SSI_civilizations_w7_ft1_nvadj_nolemma/SSI_civilizations_w7_ft1_nvadj_nolemma_co-occurrence.csv.zip SSI_civilizations_w7_ft1_nvadj_nolemma


# # Courier corpus (page)
# # Co-occurrence computations

# # 1) concept “culture”, ignore concept, adjectives, nouns, verbs, w5, ft10, nolemma, only alpha:
# co_occurrence $DEFAULT_OPTS --context-width 2 --concept "culture" --tf-threshold 10 doit.yml /data/inidun/courier_page_20210921.zip /data/inidun/shared/Courier_culture_w5_nvadj_nolemma/Courier_culture_w5_nvadj_nolemma_co-occurrence.csv.zip Courier_culture_w5_nvadj_nolemma
# # 2) concept “culture”, ignore concept, adjectives, nouns, verbs, w11, ft10, nolemma, only alpha:
# co_occurrence $DEFAULT_OPTS --context-width 5 --concept "culture" --tf-threshold 10 doit.yml /data/inidun/courier_page_20210921.zip /data/inidun/shared/Courier_culture_w11_nvadj_nolemma/Courier_culture_w11_nvadj_nolemma_co-occurrence.csv.zip Courier_culture_w11_nvadj_nolemma
# # 3) concept “cultures”, ignore concept, adjectives, nouns, verbs, w5, ft10, nolemma, only alpha:
# co_occurrence $DEFAULT_OPTS --context-width 2 --concept "cultures" --tf-threshold 10 doit.yml /data/inidun/courier_page_20210921.zip /data/inidun/shared/Courier_cultures_w5_nvadj_nolemma/Courier_cultures_w5_nvadj_nolemma_co-occurrence.csv.zip Courier_cultures_w5_nvadj_nolemma
# # 4) concept “cultures”, ignore concept, adjectives, nouns, verbs, w11, ft10, nolemma, only alpha:
# co_occurrence $DEFAULT_OPTS --context-width 5 --concept "cultures" --tf-threshold 10 doit.yml /data/inidun/courier_page_20210921.zip /data/inidun/shared/Courier_cultures_w11_nvadj_nolemma/Courier_cultures_w11_nvadj_nolemma_co-occurrence.csv.zip Courier_cultures_w11_nvadj_nolemma
# # 5) concept “civilization”, ignore concept, adjectives, nouns, verbs, w5, ft10, nolemma, only alpha:
# co_occurrence $DEFAULT_OPTS --context-width 2 --concept "civilization" --tf-threshold 10 doit.yml /data/inidun/courier_page_20210921.zip /data/inidun/shared/Courier_civilization_w5_nvadj_nolemma/Courier_civilization_w5_nvadj_nolemma_co-occurrence.csv.zip Courier_civilization_w5_nvadj_nolemma
# # 6) concept “civilization”, ignore concept, adjectives, nouns, verbs, w11, ft10, nolemma, only alpha:
# co_occurrence $DEFAULT_OPTS --context-width 5 --concept "civilization" --tf-threshold 10 doit.yml /data/inidun/courier_page_20210921.zip /data/inidun/shared/Courier_civilization_w11_nvadj_nolemma/Courier_civilization_w11_nvadj_nolemma_co-occurrence.csv.zip Courier_civilization_w11_nvadj_nolemma
# # 7) concept “civilizations”, ignore concept, adjectives, nouns, verbs, w5, ft10, nolemma, only alpha:
# co_occurrence $DEFAULT_OPTS --context-width 2 --concept "civilizations" --tf-threshold 10 doit.yml /data/inidun/courier_page_20210921.zip /data/inidun/shared/Courier_civilizations_w5_nvadj_nolemma/Courier_civilizations_w5_nvadj_nolemma_co-occurrence.csv.zip Courier_civilizations_w5_nvadj_nolemma
# # 8) concept “civilizations”, ignore concept, adjectives, nouns, verbs, w11, ft10, nolemma, only alpha:
# co_occurrence $DEFAULT_OPTS --context-width 5 --concept "civilizations" --tf-threshold 10 doit.yml /data/inidun/courier_page_20210921.zip /data/inidun/shared/Courier_civilizations_w11_nvadj_nolemma/Courier_civilizations_w11_nvadj_nolemma_co-occurrence.csv.zip Courier_civilizations_w11_nvadj_nolemma

# SSI corpus (non-regional)
# Co-occurrence computations. (Trying the four concept words with only one context window, at w=7)

# TARGET_FOLDER=./data/inidun/ssi

# mkdir -p $TARGET_FOLDER

# 1)   concept “culture”, ignore concept, adjectives, nouns, verbs, w7, ft1, nolemma, only alpha:
# co-occurrence $DEFAULT_OPTS --context-width 3 --concept "culture" $CONFIG_FILE $CORPUS_FILE $TARGET_FOLDER/SSI_culture_w7_ft1_nvadj_nolemma/SSI_culture_w7_ft1_nvadj_nolemma_co-occurrence.csv.zip
# 2)   concept “cultures”, ignore concept, adjectives, nouns, verbs, w7, ft1, nolemma, only alpha:
# co_occurrence $DEFAULT_OPTS --context-width 3 --concept "cultures" $CONFIG_FILE $CORPUS_FILE $TARGET_FOLDER/SSI_cultures_w7_ft1_nvadj_nolemma/SSI_cultures_w7_ft1_nvadj_nolemma_co-occurrence.csv.zip SSI_cultures_w7_ft1_nvadj_nolemma
# # 3)   concept “civilization”, ignore concept, adjectives, nouns, verbs, w7, ft1, nolemma, only alpha:
# co_occurrence $DEFAULT_OPTS --context-width 3 --concept "civilization" $CONFIG_FILE $CORPUS_FILE $TARGET_FOLDER/SSI_civilization_w7_ft1_nvadj_nolemma/SSI_civilization_w7_ft1_nvadj_nolemma_co-occurrence.csv.zip SSI_civilization_w7_ft1_nvadj_nolemma
# # 4)   concept “civilizations”, ignore concept, adjectives, nouns, verbs, w7, ft1, nolemma, only alpha:
# co_occurrence $DEFAULT_OPTS --context-width 3 --concept "civilizations" $CONFIG_FILE $CORPUS_FILE $TARGET_FOLDER/SSI_civilizations_w7_ft1_nvadj_nolemma/SSI_civilizations_w7_ft1_nvadj_nolemma_co-occurrence.csv.zip SSI_civilizations_w7_ft1_nvadj_nolemma

