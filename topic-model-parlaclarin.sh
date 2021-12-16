#!/bin/bash

# gensim_lda-multicore gensim_mallet-lda

N_TOPICS=200

    # --lemmatize \
    # --to-lower \
    # --remove-stopwords swedish \
    # --train-corpus-folder ./data/riksprot-speech-train-corpus \
    # --min-word-length 1 \
    # --only-any-alphanumeric \

PYTHONPATH=. python penelope/scripts/topic_model.py \
    --passthrough-column lemma \
    --n-topics 200 \
    --engine "gensim_lda-multicore" \
    --random-seed 42 \
    --alpha asymmetric \
    --max-iter 3000 \
    --store-corpus \
    --workers 4 \
    --target-folder ./data \
    ./riksprot-parlaclarin.yml \
    riksprot-parlaclarin-protokoll-200-lemma
