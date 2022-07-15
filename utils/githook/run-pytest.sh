#!/usr/bin/env bash

DIR=`git rev-parse --show-toplevel`
BRANCH=`git branch | grep '*' | sed 's/* //'`
if [ $BRANCH = "main" ]; then
    pytest
    if [ $? -ne 0 ]; then
        echo "Cannot commit/push with failing unit tests"
        exit 1
    fi
fi

if [ $BRANCH != "main" ]; then
    pytest
    if [ $? -ne 0 ]; then
        echo "#########################################################"
        echo "#########################################################"
        echo "## !!!Please fix failing unit tests before pushing!!!! ##"
        echo "#########################################################"
        echo "#########################################################"
    fi
fi

