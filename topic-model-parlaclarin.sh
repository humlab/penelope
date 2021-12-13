#!/bin/bash

# gensim_lda-multicore gensim_mallet-lda

N_TOPICS=200

PYTHONPATH=. python penelope/scripts/topic_model.py \
    --n-topics 200 \
    --lemmatize \
    --to-lower \
    --remove-stopwords swedish \
    --min-word-length 1 \
    --only-any-alphanumeric \
    --engine gensim_lda-multicore \
    --random-seed 42 \
    --alpha asymmetric \
    --max-iter 3000 \
    --store-corpus \
    --workers 4 \
    --target-folder ./data \
    --train-corpus-folder ./data/riksprot-speech-train-corpus \
    ./riksprot-parlaclarin.yml \
    riksprot-parlaclarin-protokoll-200-lemma
