#!/bin/bash
# Model name: [lang]_[capability]_[genre]_[size]
# Capability: core for general-purpose model with vocabulary, syntax, entities and word vectors, or depent for only vocab, syntax and entities)
# genre: Type of text the model is trained on (e.g. web for web text, news for news text)
# size: Model size indicator (sm, md or lg)

DOWNLOAD_URL=https://github.com/explosion/spacy-models/releases/download

target_folder="/data/lib/spacy_data"

declare -a MODELS=("en_core_web_sm" "en_core_web_md")

read -ra VERSIONS <<< $(git ls-remote --tags git@github.com:explosion/spacy-models | grep en_core_web_sm | cut -f2 | cut -d'/' -f3 | cut -d'-' -f2 | sort | tail -3 | tr '\n' ' ')

# declare -a LANGUAGES=("en")
# declare -a GENRES=("web")
# declare -a CAPABILITIES=("core")
# declare -a SIZES=("sm" "md")

#python -m nltk.downloader -d /usr/local/share/nltk_data stopwords punkt sentiwordnet

function download_model {

    local model=$1
    local version=$2

    local tarball=${model}-${version}.tar.gz
    local url=${DOWNLOAD_URL}/${model}-${version}/${tarball}

    mkdir -p $target_folder/$version
    cd $target_folder/$version

    rm -rf ${model} ${model}-${version}
    wget -qO ${tarball} ${url}
    tar xf ${tarball}
    mv ./${model}-${version}/${model}/${model}-${version} ${model}
    rm -rf ${tarball}

    echo "info: ${model}-${version} downloaded to $target_folder/$version"
}

pushd . > /dev/null

echo "models: ${MODELS[@]}"
echo "versions: ${VERSIONS[@]}"

for model in "${MODELS[@]}" ; do
    echo "info: ${model}"
    for version in "${VERSIONS[@]}" ; do
        if [ ! -d "$target_folder/$version/${model}" ]; then
            download_model $model $version
        fi
    done
done


popd > /dev/null
