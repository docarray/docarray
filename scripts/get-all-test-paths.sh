#!/usr/bin/env bash

set -ex

BATCH_SIZE=5
declare -a array1=( "tests/unit/*.py" )
declare -a array2=( $(ls -d tests/unit/*/ | grep -v '__pycache__' | grep -v 'array') )
declare -a array3=( "tests/unit/array/*.py" )
declare -a mixins=( $(find tests/unit/array/mixins -name "*.py") )
declare -a array4=( "$(echo "${mixins[@]}" | xargs -n$BATCH_SIZE)" )
dest=( "${array1[@]}" "${array2[@]}" "${array3[@]}" "${array4[@]}" )

printf '%s\n' "${dest[@]}" | jq -R . | jq -cs .
