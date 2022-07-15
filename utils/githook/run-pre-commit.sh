#!/usr/bin/env bash

set -e
echo "Running pre-commit checks ..."
DIR=`git rev-parse --show-toplevel`
source $DIR/utils/githook/run-lint-flake.sh
source $DIR/utils/githook/run-pytest.sh
source $DIR/utils/githook/fix-whitespace.sh