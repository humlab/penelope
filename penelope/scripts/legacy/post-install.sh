#!/bin/bash

export NLTK_DATA='/data/lib/nltk_data'

mkdir -p /data/lib/nltk_data

poetry run python - << EOF

import nltk
nltk.download(['punkt'])
nltk.download(['stopwords'])

EOF

