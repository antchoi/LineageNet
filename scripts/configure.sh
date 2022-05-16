#!/bin/bash
ARCH=$(uname -m)
SCRIPT_DIR=$(dirname $(readlink -f $0))
ROOT_DIR="${SCRIPT_DIR}/.."

git config --local core.autocrlf input

# Configure dependencies for runtime
pip install --upgrade pip
pip install --no-cache -r ${ROOT_DIR}/requirements.txt

pre-commit autoupdate
pre-commit install
