#!/usr/bin/env bash

set -ex

declare -a oldproto=( $(find tests -name '*oldproto*') )
dest=( "${oldproto[@]}" )

printf '%s\n' "${dest[@]}" | jq -R . | jq -cs .