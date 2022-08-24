#!/usr/bin/env bash
set -ex

# Do NOT use this directly, use jinaai/protogen image
# use jinaai/protogen:3.21 in order to use compiler version == 21 (creates pb/docarray_pb2.py)
# and use jinaai/protogen:latest to use compiler version <= 20 (creates pb2/docarray_pb2.py)
# make sure to use jinaai/protogen:3.21 to avoid overriting the module
#
# current dir: docarray/docarray
# run the following in bash:
# docker run -v $(pwd)/proto:/jina/proto jinaai/protogen
# finally, set back owner of the generated files using: sudo chown -R $(id -u ${USER}):$(id -g ${USER}) .

SRC_DIR=./
MODULE=docarray
SRC_NAME="${MODULE}.proto"
COMP_OUT_NAME="${MODULE}_pb2.py"

OUT_FOLDER="${2:-pb2}/"

VER_FILE=../__init__.py

if [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; then
    echo "Error: Please specify the [PATH_TO_GRPC_PYTHON_PLUGIN], refer more details at " \
      "https://docarray.jina.ai/"
    printf "\n"
    echo "USAGE:"
    printf "\t"
    echo "bash ./build-proto.sh [PATH_TO_GRPC_PYTHON_PLUGIN]"
    exit 1
fi

PLUGIN_PATH=${1}  # /Volumes/TOSHIBA-4T/Documents/grpc/bins/opt/grpc_python_plugin

printf "\e[1;33mgenerating protobuf and grpc python interface\e[0m\n"

protoc -I ${SRC_DIR} --python_out="${SRC_DIR}${OUT_FOLDER}" ${SRC_DIR}${SRC_NAME}
