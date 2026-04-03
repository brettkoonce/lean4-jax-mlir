#!/bin/bash
set -e

mkdir -p data
cd data

BASE="https://storage.googleapis.com/cvdf-datasets/mnist"

for FILE in train-images-idx3-ubyte.gz \
            train-labels-idx1-ubyte.gz \
            t10k-images-idx3-ubyte.gz  \
            t10k-labels-idx1-ubyte.gz; do
  if [ ! -f "${FILE%.gz}" ]; then
    echo "Downloading $FILE ..."
    curl -O "$BASE/$FILE"
    gunzip -f "$FILE"
  else
    echo "${FILE%.gz} already exists, skipping."
  fi
done

echo "Done. Files in ./data/"
ls -lh
