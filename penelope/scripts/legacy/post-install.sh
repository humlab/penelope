#!/bin/bash

export NLTK_HOME='/data/vendor/nltk_home'

mkdir -p /data/vendor/nltk_home

poetry run python - << EOF

import nltk
nltk.download('punkt'])
nltk.download('stopwords'])

EOF

