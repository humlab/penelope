#!/bin/bash

# gensim_lda-multicore gensim_mallet-lda

N_TOPICS=500

    # --lemmatize \
    # --to-lower \
    # --remove-stopwords swedish \
    # --train-corpus-folder ./data/riksprot-speech-train-corpus \
    # --min-word-length 1 \
    # --only-any-alphanumeric \
    # --random-seed 42 \

PYTHONPATH=. python penelope/tm/train.py \
    --passthrough-column lemma \
    --n-topics 500 \
    --engine "gensim_lda-multicore" \
    --alpha asymmetric \
    --max-iter 3000 \
    --store-corpus \
    --workers 4 \
    --target-folder ./data \
    /data/riksdagen_corpus_data/tagged-speech-corpus.v0.3.0.id.1960/corpus.yml \
    riksprot-parlaclarin-1960-protokoll-500-lemma
