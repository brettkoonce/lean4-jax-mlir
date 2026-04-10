#!/bin/bash
set -euo pipefail

# Download CIFAR-10 binary version
# Creates data/ directory with the 6 batch files

DIR="data"
URL="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
ARCHIVE="cifar-10-binary.tar.gz"

mkdir -p "$DIR"

echo "Downloading CIFAR-10 binary dataset..."
curl -L -o "$ARCHIVE" "$URL"

echo "Extracting..."
tar xzf "$ARCHIVE" -C "$DIR" --strip-components=1

echo "Verifying files..."
for f in data_batch_1.bin data_batch_2.bin data_batch_3.bin data_batch_4.bin data_batch_5.bin test_batch.bin; do
  if [ ! -f "$DIR/$f" ]; then
    echo "ERROR: $DIR/$f not found!"
    exit 1
  fi
  SIZE=$(stat -c%s "$DIR/$f" 2>/dev/null || stat -f%z "$DIR/$f")
  echo "  $f  ($SIZE bytes)"
done

rm -f "$ARCHIVE"
echo "Done! Data is in ./$DIR/"
