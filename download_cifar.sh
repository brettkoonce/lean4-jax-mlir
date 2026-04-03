#!/bin/bash
set -e
mkdir -p data/cifar-10
cd data/cifar-10
URL="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
if [ ! -f "data_batch_1.bin" ]; then
  echo "Downloading CIFAR-10..."
  curl -O "$URL"
  tar xzf cifar-10-binary.tar.gz --strip-components=1
  rm cifar-10-binary.tar.gz
fi
echo "Done. Files in ./data/cifar-10/"
ls -lh *.bin
