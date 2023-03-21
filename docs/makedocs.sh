#!/usr/bin/env bash

cp ../README.md .
cp ../CONTRIBUTING.md .

cd ..
poetry run mkdocs build