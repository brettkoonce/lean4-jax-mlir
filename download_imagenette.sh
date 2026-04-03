#!/bin/bash
set -e
mkdir -p data/imagenette
cd data/imagenette
if [ ! -f "train.bin" ]; then
  echo "Downloading imagenette2-320..."
  curl -LO https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
  echo "Extracting..."
  tar xzf imagenette2-320.tgz
  echo "Preprocessing to binary format (requires: pip install Pillow)..."
  python3 ../../preprocess_imagenette.py imagenette2-320 .
  rm -rf imagenette2-320 imagenette2-320.tgz
fi
echo "Done. Files in ./data/imagenette/"
ls -lh *.bin
