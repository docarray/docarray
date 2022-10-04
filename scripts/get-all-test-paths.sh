#!/usr/bin/env bash

set -ex

BATCH_SIZE=3
declare -a mixins=( $(find tests -name "test_*.py" -not -path '*/paddle/*') )
declare -a array4=( "$(echo "${mixins[@]}" | xargs -n$BATCH_SIZE)" )
declare -a array5=( $(ls -d tests/unit/array/*/ | grep -v '__pycache__' | grep -v 'mixins' | grep -v 'paddle') )
dest=( "${array1[@]}" "${array2[@]}" "${array3[@]}" "${array4[@]}" "${array5[@]}" )

printf '%s\n' "${dest[@]}" | jq -R . | jq -cs .