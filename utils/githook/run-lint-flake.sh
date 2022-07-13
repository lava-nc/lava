#!/usr/bin/env bash

flakeheaven lint src/lava tests

if [ $? -ne 0 ]; then
    echo "flake check failed"
    return 1
fi
