#!/bin/bash

# ONLY NEEDED IN CI

# Get the input variable
input_variable=$1

# Check if the input variable is "true"
if [ "$input_variable" == "pydantic-v2" ]; then
  echo "Installing or updating pydantic..."
  poetry run pip install -U pydantic
else
  echo "Skipping installation of pydantic."
fi
