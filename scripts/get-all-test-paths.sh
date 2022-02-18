#!/usr/bin/env bash

set -ex

declare -a array1=( "tests/unit/*.py")
declare -a array2=( $(ls -d tests/unit/*/ | grep -v '__pycache__' ))
dest=( "${array1[@]}" "${array2[@]}" )

printf '%s\n' "${dest[@]}" | jq -R . | jq -cs .
