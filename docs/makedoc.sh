#!/usr/bin/env bash

set -ex

rm -rf api && make clean

docker run --rm \
  -v $(pwd)/proto:/out \
  -v $(pwd)/../docarray/proto:/protos \
  pseudomuto/protoc-gen-doc --doc_opt=markdown,docs.md

cp ../README.md .
cp ../CONTRIBUTING.md.md .


cat README.md index_init.md > index.md
make dirhtml
