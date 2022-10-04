#!/usr/bin/env bash

set -ex

BATCH_SIZE=3
declare -a paddle=( $(find tests -name '*paddle*') )
dest=( "${paddle[@]}" )

printf '%s\n' "${dest[@]}" | jq -R . | jq -cs .