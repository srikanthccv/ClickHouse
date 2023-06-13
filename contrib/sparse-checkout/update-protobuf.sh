#!/bin/sh

echo "Using sparse checkout for protobuf"

FILES_TO_CHECKOUT=$(git rev-parse --git-dir)/info/sparse-checkout
echo '!/*' > $FILES_TO_CHECKOUT
echo '/*/*' >> $FILES_TO_CHECKOUT
echo '/src/*' >> $FILES_TO_CHECKOUT
echo '/cmake/*' >> $FILES_TO_CHECKOUT

git config core.sparsecheckout true
git checkout $1
git read-tree -mu HEAD
