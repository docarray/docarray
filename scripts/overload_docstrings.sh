#!/bin/bash

# this script is run by jina-dev-bot

# fix overload doccstrings && black it
arr=( $(python overload_docstrings.py) ) && echo "${arr}" && black -S "${arr[@]}"