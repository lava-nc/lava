#!/usr/bin/env bash

DIR=`git rev-parse --show-toplevel`
ln -s $DIR/utils/githook/run-pre-commit.sh $DIR/.git/hooks/pre-commit
